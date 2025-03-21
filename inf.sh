# NOTE: use max_images = 1 to debug each one
# NOTE: run this in separete terminals

# outputs_3_15 => depth map for whole image (TODO: delete later)
# outputs_3_18 => depth map for fg only (TODO: delete later)
# outputs_3_19 => i put all result in two folders with different surfix

run_inference() {
    # Check that at least two arguments are provided
    if [ "$#" -lt 2 ]; then
        echo "Usage: run_inference <relight_type> <gpu> [depth_fg_only_flag]"
        return 1
    fi

    local relight_type="$1"
    local gpu="$2"
    local depth_flag="$3"  # If non-empty, include --depth_fg_only

    # Fixed parameters
    local input_dir="/home/shenzhen/Datasets/dataset_with_garment_debug_100"
    local output_dir="outputs_3_19"

    # Build the command
    # NOTE: use max_images = 1 for debug. remove later!!! 
    local cmd="python inf.py \
        --input_dir ${input_dir} \
        --output_dir ${output_dir} \
        --relight_type ${relight_type} \
        --gpu ${gpu}"
    
    # Optionally add the depth flag if depth_flag is "true"
    if [ "${depth_flag}" = "true" ]; then
        cmd+=" --depth_fg_only"
    fi

    echo "Executing: ${cmd}"
    eval "${cmd}"
}

# # # Example usage:
# run_inference golden_hour 0 false &
# run_inference moonlight 1 false &
# run_inference noon_sunlight 2 false &
# run_inference neon_lights 3 false &
# run_inference candlelight 4 false &
# run_inference spotlight 5 false &
# run_inference thunderstorm 6 false &
# run_inference meteor_shower 7 false &
# wait

# run_inference golden_hour 0 true &
# run_inference moonlight 1 true &
# run_inference noon_sunlight 2 true &
# run_inference neon_lights 3 true &
# run_inference candlelight 4 true &
# run_inference spotlight 5 true &
# run_inference thunderstorm 6 true &
# run_inference meteor_shower 7 true &
# wait

# run_inference volcanic_glow 0 false &
# run_inference foggy_morning 1 false &
# run_inference volcanic_glow 2 true &
# run_inference foggy_morning 3 true &
# wait


# # More sophisticated example usage:
# run_inference golden_hour_2 0 false &
# run_inference moonlight_2 1 false &
# run_inference noon_sunlight_2 2 false &
# run_inference neon_lights_2 3 false &
# run_inference candlelight_2 4 false &
# run_inference spotlight_2 5 false &
# run_inference thunderstorm_2 6 false &
# run_inference meteor_shower_2 7 false &
# wait


# run_inference golden_hour_2 0 true &
# run_inference moonlight_2 1 true &
# run_inference noon_sunlight_2 2 true &
# run_inference neon_lights_2 3 true &
# run_inference candlelight_2 4 true &
# run_inference spotlight_2 5 true &
# run_inference thunderstorm_2 6 true &
# run_inference meteor_shower_2 7 true &
# wait

# run_inference volcanic_glow_2 0 false &
# run_inference foggy_morning_2 1 false &
# run_inference volcanic_glow_2 2 true &
# run_inference foggy_morning_2 3 true &
# wait


# run_inference golden_hour_2 0 true &
# run_inference moonlight_2 1 true &
# run_inference noon_sunlight_2 2 true &
# run_inference neon_lights_2 3 true &
# run_inference candlelight_2 5 true &
# run_inference spotlight_2 6 true &
# run_inference thunderstorm_2 7 true &
# wait

# run_inference volcanic_glow_2 0 false &
# run_inference foggy_morning_2 1 false &
# run_inference volcanic_glow_2 2 true &
# run_inference foggy_morning_2 3 true &
# run_inference meteor_shower_2 5 true &
# wait


# run_inference volcanic_glow_2 0 false &
# run_inference foggy_morning_2 1 false &
# run_inference volcanic_glow_2 2 true &
# run_inference foggy_morning_2 3 true &
# run_inference meteor_shower_2 4 true &
# wait
