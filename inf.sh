# exp_3_19      => Baseline (just do it)
# exp_3_21      => Force output to match input shape (terrible results)
# exp_3_21_v2   => Foreground depth only; background and OOB areas set to zero depth
# exp_3_21_v3   => Reshape output of exp_3_19 to match input shape
# exp_3_28_v3 => exp_3_21_v3 + relighting_prompts_2

# copy_fg_depth experiments
# outpaint: outpaint_extend_6 => inference: exp_3_28_v1/2 (final+prompt_2: false/true)

# # NOTE: match source resolution exps
# python inf.py --exp_config configs/exp_3_28_v3.yaml --relight_type golden_hour --gpu 1

# python inf.py --exp_config configs/exp_3_28_v3.yaml --relight_type moonlight --gpu 0
# python inf.py --exp_config configs/exp_3_28_v3.yaml --relight_type noon_sunlight --gpu 0
# python inf.py --exp_config configs/exp_3_28_v3.yaml --relight_type neon_lights --gpu 0
# python inf.py --exp_config configs/exp_3_28_v3.yaml --relight_type candlelight --gpu 0
# python inf.py --exp_config configs/exp_3_28_v3.yaml --relight_type spotlight --gpu 0
# python inf.py --exp_config configs/exp_3_28_v3.yaml --relight_type thunderstorm --gpu 0

# python outpaint2.py --exp_config configs/exp_3_31_v1.yaml --relight_type golden_hour --gpu 0
# python outpaint2.py --exp_config configs/exp_3_31_v1.yaml --relight_type moonlight --gpu 2

# python outpaint2.py --exp_config configs/exp_3_31_v2.yaml --relight_type golden_hour --gpu 1
# python outpaint2.py --exp_config configs/exp_3_31_v2.yaml --relight_type moonlight --gpu 3

# NOTE: this is currently the best result
# python outpaint2.py --exp_config configs/exp_3_31_v3.yaml --relight_type golden_hour --gpu 1
# python outpaint2.py --exp_config configs/exp_3_31_v3.yaml --relight_type moonlight --gpu 2
# python outpaint2.py --exp_config configs/exp_3_31_v3.yaml --relight_type neon_lights --gpu 3
# python outpaint2.py --exp_config configs/exp_3_31_v3.yaml --relight_type candlelight --gpu 4
# python outpaint2.py --exp_config configs/exp_3_31_v3.yaml --relight_type spotlight --gpu 5
# python outpaint2.py --exp_config configs/exp_3_31_v3.yaml --relight_type foggy_morning --gpu 6

# python outpaint2.py --exp_config configs/exp_3_31_v4.yaml --relight_type golden_hour --gpu 0

# python outpaint2.py --exp_config configs/exp_3_31_v5.yaml --relight_type golden_hour --gpu 0
# python outpaint2.py --exp_config configs/exp_3_31_v6.yaml --relight_type golden_hour --gpu 0

# python outpaint2.py --exp_config configs/exp_4_2_v1.yaml --relight_type golden_hour --gpu 0
# python outpaint2.py --exp_config configs/exp_4_2_v1.yaml --relight_type neon_lights --gpu 2
# python outpaint2.py --exp_config configs/exp_4_2_v1.yaml --relight_type candlelight --gpu 3
# python outpaint2.py --exp_config configs/exp_4_2_v1.yaml --relight_type foggy_morning --gpu 5


# python outpaint2.py --exp_config configs/exp_4_3_v1.yaml --relight_type golden_hour --gpu 1
# python outpaint2.py --exp_config configs/exp_4_3_v1.yaml --relight_type noon_sunlight --gpu 0
# python outpaint2.py --exp_config configs/exp_4_3_v1.yaml --relight_type neon_lights --gpu 0
# python outpaint2.py --exp_config configs/exp_4_3_v1.yaml --relight_type candlelight --gpu 0
# python outpaint2.py --exp_config configs/exp_4_3_v1.yaml --relight_type foggy_morning --gpu 1


# python outpaint2.py --exp_config configs/exp_4_3_v1.yaml --relight_type spotlight --gpu 0
# python outpaint2.py --exp_config configs/exp_4_3_v1.yaml --relight_type moonlight --gpu 0
# python outpaint2.py --exp_config configs/exp_4_3_v1.yaml --relight_type snowy_morning --gpu 1


# Later: try even higher resolutions: 896, 1008, 672 (let's use golden_hour to save time )
# python outpaint2.py --exp_config configs/exp_4_3_v2.yaml --relight_type golden_hour --gpu 0 # (896) looks similar to resolution of 784, but lighting slight more unrealistic
# python outpaint2.py --exp_config configs/exp_4_3_v3.yaml --relight_type golden_hour --gpu 1 # (1008) very bad lighting, takes longer inference time
# python outpaint2.py --exp_config configs/exp_4_3_v4.yaml --relight_type golden_hour --gpu 0 # (672) facial details are not good 

# exp_4_3_v2 (repeat this experiment, with all relighting types)
# python outpaint2.py --exp_config configs/exp_4_3_v2.yaml --relight_type noon_sunlight --gpu 1
# python outpaint2.py --exp_config configs/exp_4_3_v2.yaml --relight_type neon_lights --gpu 2
# python outpaint2.py --exp_config configs/exp_4_3_v2.yaml --relight_type candlelight --gpu 3
# python outpaint2.py --exp_config configs/exp_4_3_v2.yaml --relight_type foggy_morning --gpu 4
# python outpaint2.py --exp_config configs/exp_4_3_v2.yaml --relight_type moonlight --gpu 0

# TODO: think about more/better relighting types

# NOTE: if train on entire dataset, first copy to /scratch, /ssd, or /ssd1