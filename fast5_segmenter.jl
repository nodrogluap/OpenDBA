# Load Oxford Nanopore Technologies FAST5 files, do outlier value cleanup/averaging, then segment the results according to expected translocation rate

# Segmentation based on http://homepages.spa.umn.edu/~willmert/science/ksegments/, updated to Julia 1.1 libraries, data structures and syntax, 
# and writing the data to files.
# Paul Gordon, 2019 (gordonp@ucalgary.ca)

# Uncomment the following lines the first time that you run the program, to ensure you have support for HDF5 file reading and some basic stats. 
# Automatic plotting of the segmentation is disabled at the moment.
#import Pkg
#Pkg.add("HDF5")
#Pkg.add("FreqTables");
#Pkg.add("Statistics");
#Pkg.add("StatsBase");
#Pkg.add("Plots")
#Pkg.add("PyPlot")
#using Plots
#pyplot()
using HDF5
using FreqTables
using Statistics
using StatsBase
using LinearAlgebra
using DelimitedFiles

function prepare_ksegments(series::Array{Int16,1}, weights::Array{Float64,1})
    N = length(series);

    # Pre-allocate matrices
    wgts = diagm(0 => weights);
    wsum = diagm(0 => weights .* series);
    sqrs = diagm(0 => weights .* series .* series);

    # Also initialize the outputs with sane defaults
    dists = zeros(Float64, N,N);
    means = diagm(0 => convert(Array{Float64}, series));

    # Fill the upper triangle of dists and means by performing up-right
    # diagonal sweeps through the matrices
    for δ=1:N
        for l=1:(N-δ)
            # l = left boundary, r = right boundary
            r = l + δ;

            # Incrementally update every partial sum
            wgts[l,r] = wgts[l,r-1] + wgts[r,r];
            wsum[l,r] = wsum[l,r-1] + wsum[r,r];
            sqrs[l,r] = sqrs[l,r-1] + sqrs[r,r];

            # Calculate the mean over the range
            means[l,r] = wsum[l,r] / wgts[l,r];
            # Then update the distance calculation. Normally this would have a term
            # of the form
            #   - wsum[l,r].^2 / wgts[l,r]
            # but one of the factors has already been calculated in the mean, so
            # just reuse that.
            dists[l,r] = sqrs[l,r] - means[l,r]*wsum[l,r];        
        end
    end

    return (dists,means)
end

function regress_ksegments(series::Array{Int16,1}, weights::Array{Float64,1}, k::Int)

    # Make sure we have a row vector to work with
    if (length(series) == 1)
        # Only a scalar value
        error("series must have length > 1")
    end

    # Ensure series and weights have the same size
    if (size(series) != size(weights))
        error("series and weights must have the same shape")
    end

    # Make sure the choice of k makes sense
    if (k < 1 || k > length(series))
        error("k must be in the range 1 to length(series)")
    end

    N = length(series);

    # Get pre-computed distances and means for single-segment spans over any
    # arbitrary subsequence series(i:j). The costs for these subsequences will
    # be used *many* times over, so a huge computational factor is saved by
    # just storing these ahead of time.
    (one_seg_dist,one_seg_mean) = prepare_ksegments(series, weights);

    # Keep a matrix of the total segmentation costs for any p-segmentation of
    # a subsequence series[1:n] where 1<=p<=k and 1<=n<=N. The extra column at
    # the beginning is an effective zero-th row which allows us to index to
    # the case that a (k-1)-segmentation is actually disfavored to the 
    # whole-segment average.
    k_seg_dist = zeros(Float64, k, N+1);
    # Also store a pointer structure which will allow reconstruction of the
    # regression which matches. (Without this information, we'd only have the
    # cost of the regression.)
    k_seg_path = zeros(Int, k, N);

    # Initialize the case k=1 directly from the pre-computed distances
    k_seg_dist[1,2:end] = one_seg_dist[1,:];

    # Any path with only a single segment has a right (non-inclusive) boundary
    # at the zeroth element.
    for i=1:N
        k_seg_path[1,i] = 0;
    end
    # Then for p segments through p elements, the right boundary for the (p-1)
    # case must obviously be (p-1).
    for i in 1:k
        k_seg_path[i,i] = k - 1;
    end

    # Now go through all remaining subcases 1 < p <= k
    for p=2:k
        # Update the substructure as successively longer subsequences are
        # considered.
        for n=p:N
            # Enumerate the choices and pick the best one. Encodes the recursion
            # for even the case where j=1 by adding an extra boundary column on the
            # left side of k_seg_dist. The j-1 indexing is then correct without
            # subtracting by one since the real values need a plus one correction.
            choices = Array{Float64}(undef, n);
            for i=1:n
                choices[i] = k_seg_dist[p-1, i] + one_seg_dist[i, n];
            end

            (bestval,bestidx) = findmin(choices);

            # Store the sub-problem solution. For the path, store where the (p-1)
            # case's right boundary is located.
            k_seg_path[p,n] = bestidx - 1;
            # Then remember to offset the distance information due to the boundary
            # (ghost) cells in the first column.
            k_seg_dist[p,n+1] = bestval;
        end
    end

    # Eventual complete regression
    reg = zeros(Float64, size(series));

    # Now use the solution information to reconstruct the optimal regression.
    # Fill in each segment reg(i:j) in pieces, starting from the end where the
    # solution is known.
    rhs = length(reg);
    for p=k:-1:1
        # Get the corresponding previous boundary
        lhs = k_seg_path[p,rhs];

        # The pair (lhs,rhs] is now a half-open interval, so set it appropriately
        for i=lhs+1:rhs
            reg[i] = one_seg_mean[lhs+1,rhs];
        end

        # Update the right edge pointer
        rhs = lhs;
    end

    return reg
end

translocation_rate_per_second = parse(Int64, ARGS[1]);
sampling_frequency = parse(Int64, ARGS[2]);
max_samples_to_segment = parse(Int64, ARGS[3]);
max_samples_to_supersegment = parse(Int64, ARGS[4]);
output_prefix = ARGS[5];

avg_segment_size = 1000/translocation_rate_per_second; # in ms
#println("Average expected segment size was ", avg_segment_size);

# read the fast5 file name from the command line
for argi=6:size(ARGS)[1]
    fast5 = h5open(ARGS[argi], "r");
    channel = fast5["UniqueGlobalKey/channel_id"];
    channel_number = read(attrs(channel), "channel_number");
    for nanopore_read in fast5["/Raw/Reads"]
        whole_raw_signal = read(nanopore_read, "Signal");
        read_number = read(attrs(nanopore_read), "read_number");
        output_file_prefix = joinpath(output_prefix,string("ch", channel_number, "_read", read_number));

        writedlm(string(output_file_prefix, ".raw.txt"), whole_raw_signal, "\n");
	# Commented lines represent option to print first pass segemtnation results to a file for plotting or debugging.
        # fileName = string(output_file_prefix, ".max", translocation_rate_per_second, "bps.event_means.txt");
        # io = open(fileName,"w");
        fileName = string(output_file_prefix, ".max", translocation_rate_per_second, "bps.event_medians.txt");
        #io2 = open(fileName,"w");
        io = open(fileName,"w");

        num_samples = size(whole_raw_signal)[1];
        println("Processing ", output_file_prefix, " (", num_samples, " samples, ", max_samples_to_segment, " at a time)");
        first_round_medians = zeros(0);
        events_so_far = 0;
        for start_index=1:max_samples_to_segment:(num_samples-4)
            samples_to_segment = max_samples_to_segment;
            if(start_index+samples_to_segment > num_samples)
                 samples_to_segment = num_samples-start_index;
            end
            raw_signal = whole_raw_signal[start_index:(start_index+samples_to_segment)];
        #if(num_samples > max_samples_to_segment)
        #    num_samples = max_samples_to_segment;
        #    raw_signal = raw_signal[start_index:(start_index+num_samples)];
        #end
            
            # Even weighting for all data points in the series...TODO: put less weight on the starting values?
            #subsampled_length = floor(Int, num_samples/2);
            subsampled_length = floor(Int, samples_to_segment/2);
            if(subsampled_length < 3)
               continue
            end
            smoothed_signal = zeros(Int16, subsampled_length);

            # Two stage smoothing to reduce effect of high measurements (min) while tempering low measurements a bit too (mean)
            [smoothed_signal[Int(i/2)]=floor(Int16, Statistics.mean(raw_signal[(i-1):i])) for i=2:2:samples_to_segment]
            [smoothed_signal[Int(i/2)]=floor(Int16, minimum(smoothed_signal[(i-1):i])) for i=2:2:subsampled_length]
            subsampled_length = floor(Int, subsampled_length/2);
            smoothed_signal = smoothed_signal[1:subsampled_length];;
            wght = ones(subsampled_length);

            last_expected_num_events = 0;
            for test_translocation_rate_per_second=translocation_rate_per_second:-1:4
                #elapsed_time = num_samples/sampling_frequency*1000; # in milliseconds
                elapsed_time = samples_to_segment/sampling_frequency*1000; # in milliseconds
                time_scale = 0:(1/sampling_frequency*1000):elapsed_time;
                expected_num_events = ceil(Int, elapsed_time*test_translocation_rate_per_second/1000);

                if(expected_num_events > subsampled_length/2)
                    continue
                end

                # Would yield same result
                if(last_expected_num_events == expected_num_events)
                    continue
                end
                last_expected_num_events = expected_num_events

                # Run the regression
                means = regress_ksegments(smoothed_signal, wght, expected_num_events);

                # If we are getting events with a tiny number of members, let's assume have too many segments
                # Eliminate segments of size 1
                num_singletons = 0;
                for i=2:(subsampled_length-1)
                    # singleton
                    if(means[i-1] != means[i] && means[i+1] != means[i])
                        num_singletons = num_singletons + 1
                        # Pick the neighbour with the smallest distance to join
                        if(abs(means[i-1] - means[i]) < abs(means[i+1] - means[i]))
                            means[i] = means[i-1];
                        else
                            means[i] = means[i+1];
                        end
                    end
                end
                if(num_singletons > 0)
                    continue
                end
            
                # Find median for each segment
                medians = zeros(Int16, subsampled_length);
                segment_raw_values = raw_signal[1:4];
                segment_start = 1;
                for i=2:subsampled_length
                    if(means[i-1] != means[i])
                        med = floor(Int16, median(segment_raw_values));
                        # assign to all the members of the segment
                        medians[segment_start:(i-1)] .= med;

                        segment_raw_values = zeros(Int16, 0);
                        segment_start = i;
                    else
                        append!(segment_raw_values, raw_signal[(4*i-3):(4*i)]);
                    end
                end
                # Unimodal without the singletons?
                if(length(segment_raw_values) == 0)
                    continue
                end
                medians[segment_start:subsampled_length] .= floor(Int16, median(segment_raw_values));

                # println("Raw position ", start_index, " (event ", events_so_far, "), estimated rate of ", test_translocation_rate_per_second);
                events_so_far += expected_num_events-num_singletons;
                # Expand the means result to the original data length
                #means = StatsBase.inverse_rle(means, fill(4, subsampled_length));
                medians = StatsBase.inverse_rle(medians, fill(4, subsampled_length));

                # Append the regression breakpoints to a file
                append!(first_round_medians, medians);

                # Plot the raw data
                #Plots.scatter(time_scale, raw_signal,
                #  title="Unimodal regression (dwells between \noligonucleotide nanopore translocation)\n with uniform weighting",
                #  xlabel="Elapsed Time (ms)",
                #  ylabel="Electrical Current (pA)",
                #  xlim=[0,elapsed_time],
                #  xticks = Int.(round.(0:avg_segment_size:elapsed_time)),
                #  size = (plot_width, plot_height),
                #  label=""); # disable legend

                # Overlay (indicated by '!') the optimal unimodal regression
                #plot!(time_scale, means, color="red", linewidth=2, linetype=:steppre, label="")
                #plot!(time_scale, medians, color="red", linewidth=2, linetype=:steppre, label="")

                break;
            end
        end
        # writedlm(io, first_round_medians, "\n");

        # Perform a second round of segmentation over larger areas after adjusting all the original data towards the segmented medians.
        # This will reduce segmentation window edge artefacts.

        events_so_far = 0;
        final_medians = zeros(0);
        for start_index=1:max_samples_to_supersegment:(num_samples-2)
            samples_to_segment = max_samples_to_supersegment;
            if(start_index+samples_to_segment > num_samples)
                 samples_to_segment = num_samples-start_index;
            end
            raw_signal = whole_raw_signal[start_index:(start_index+samples_to_segment)];

            subsampled_length = floor(Int, samples_to_segment/2);
            if(subsampled_length < 2)
               continue
            end
            smoothed_signal = zeros(Int16, subsampled_length);

            # Single stage 2-datapoint smoothing with regional median correction to reduce wandering drift effect on oversegmentation
            [smoothed_signal[Int(i/2)]=floor(Int16, (first_round_medians[i-1]+first_round_medians[i]+raw_signal[i-1]+raw_signal[i])/4) for i=2:2:samples_to_segment]
            smoothed_signal = smoothed_signal[1:subsampled_length];
            wght = ones(subsampled_length);

            last_expected_num_events = 0;
            for test_translocation_rate_per_second=translocation_rate_per_second:-1:4
                elapsed_time = samples_to_segment/sampling_frequency*1000; # in milliseconds
                time_scale = 0:(1/sampling_frequency*1000):elapsed_time;
                expected_num_events = ceil(Int, elapsed_time*test_translocation_rate_per_second/1000);

                if(expected_num_events > subsampled_length/2)
                    continue
                end

                # Would yield same result
                if(last_expected_num_events == expected_num_events)
                    continue
                end
                last_expected_num_events = expected_num_events

                # Run the regression
                means = regress_ksegments(smoothed_signal, wght, expected_num_events);

                # If we are getting events with a tiny number of members, let's assume have too many segments
                # Eliminate segments of size 1
                num_singletons = 0;
                for i=2:(subsampled_length-1)
                    # singleton
                    if(means[i-1] != means[i] && means[i+1] != means[i])
                        num_singletons = num_singletons + 1
                        # Pick the neighbour with the smallest distance to join
                        if(abs(means[i-1] - means[i]) < abs(means[i+1] - means[i]))
                            means[i] = means[i-1];
                        else
                            means[i] = means[i+1];
                        end
                    end
                end
                if(num_singletons > 0)
                    continue
                end

                # Find median for each segment
                medians = zeros(Int16, subsampled_length);
                segment_raw_values = raw_signal[1:2];
                segment_start = 1;
                for i=2:subsampled_length
                    if(means[i-1] != means[i])
                        med = floor(Int16, median(segment_raw_values));
                        # assign to all the members of the segment
                        medians[segment_start:(i-1)] .= med;

                        segment_raw_values = zeros(Int16, 0);
                        segment_start = i;
                    else
                        append!(segment_raw_values, raw_signal[(2*i-1):(2*i)]);
                    end
                end
                # Unimodal without the singletons?
                if(length(segment_raw_values) == 0)
                    continue
                end
                medians[segment_start:subsampled_length] .= floor(Int16, median(segment_raw_values));

                println("Raw position ", start_index, " (final event ", events_so_far, "), estimated rate of ", test_translocation_rate_per_second);
                events_so_far += expected_num_events-num_singletons;
                # Expand the means result to the original data length (e.g. for plotting vs. raw)
                # medians = StatsBase.inverse_rle(medians, fill(2, subsampled_length));
                append!(final_medians, medians);
                break;
            end
        end
        #writedlm(io2, final_medians, "\n");
        writedlm(io, final_medians, "\n");
                
        # Save the plot to a file
        # Plots.savefig(string(output_file_prefix, ".max", translocation_rate_per_second, "bps.segmented.png"));

    end
end

