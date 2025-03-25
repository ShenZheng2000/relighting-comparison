# # NOTE: match source resolution exps
# python inf.py --exp_config configs/exp_3_21.yaml --relight_type golden_hour --gpu 0 &
# python inf.py --exp_config configs/exp_3_21.yaml --relight_type moonlight --gpu 1 &
# python inf.py --exp_config configs/exp_3_21.yaml --relight_type noon_sunlight --gpu 2 &
# python inf.py --exp_config configs/exp_3_21.yaml --relight_type neon_lights --gpu 3 &
# python inf.py --exp_config configs/exp_3_21.yaml --relight_type candlelight --gpu 5 &
# python inf.py --exp_config configs/exp_3_21.yaml --relight_type spotlight --gpu 6 &
# wait


# python inf.py --exp_config configs/exp_3_21.yaml --relight_type thunderstorm --gpu 0 &
# python inf.py --exp_config configs/exp_3_21.yaml --relight_type meteor_shower --gpu 1 &
# python inf.py --exp_config configs/exp_3_21.yaml --relight_type volcanic_glow --gpu 4 &
# python inf.py --exp_config configs/exp_3_21.yaml --relight_type foggy_morning --gpu 5 &
# wait

# exp_3_19      => Baseline (just do it)
# exp_3_21      => Force output to match input shape (terrible results)
# exp_3_21_v2   => Foreground depth only; background and OOB areas set to zero depth
# exp_3_21_v3   => Reshape output of exp_3_19 to match input shape