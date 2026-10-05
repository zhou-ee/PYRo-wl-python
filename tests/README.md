# 功能与数值回归验证

保留地面动力学、增益调度、腾空守恒/形状稳定性、RDOB 设计/调度、float32/C++ 对照、饱和和接触测试；增加共享参数、缓存切换、依赖边界和路径验证。

迁移前有 70 项通过、2 项失败，原因和修复记录见 `docs/migration_report.md`。目前所有物理数值、控制响应、拟合和 C++ 对照测试显式使用 `reference_parameters.py` 中独立的固定参数样本，包括质量、惯量、几何、限幅、Q/R、采样周期和接触参数。修改实际 config 文件无需更新这些数值基准；该样本不参与生产模型运行。

`test_workspace_architecture.py` 和 CLI 路径测试保留当前配置的集成检查，验证共享参数、缓存、配置传递与默认行为。现有 RDOB 重基准化测试也核对限幅取自调度自身保存的硬件参数。配置合法性或实际控制性能问题仍由模型/生成命令的验证报告反映，不通过修改固定数值基准掩盖。

## 依赖边界

pytest 读取 src 布局；C++ 对照在可用 g++ 时执行。

本次解耦验证：当前配置与仅修改 config 的另一组配置均为 90 项通过。

## 使用示例

```powershell
# PowerShell
python -m pytest -p no:cacheprovider -q
```

## 文件与接口

| 文件 | 主要接口 |
|---|---|
| [test_airborne_lqr.py](test_airborne_lqr.py) | `test_free_flight_mass_and_internal_torque_invariant`, `test_global_orientation_does_not_change_free_flight_mass`, `test_forward_body_com_makes_leg_force_pitch_coupling`, `test_only_four_shape_states_can_be_stabilized`, `test_airborne_q_shifts_pitch_leg_compromise`, `test_ground_weights_cannot_change_airborne_gain`, `coarse_schedule`, `test_float32_schedule_covers_asymmetric_lengths`, `test_export_contains_only_airborne_states_and_hip_inputs`, `test_schedule_requires_enough_nodes` |
| [test_leg_length_schedule.py](test_leg_length_schedule.py) | `test_active_u0_subtracts_gas_spring_without_changing_model`, `test_single_node_is_controllable_and_dlqr_stable`, `test_gain_schedule_fit_accuracy_and_float32_stability`, `test_left_right_symmetry_at_equal_leg_lengths`, `test_header_rendering_format` |
| [test_numeric_model.py](test_numeric_model.py) | `test_gas_spring_force_fit_matches_firmware`, `test_lbar_is_the_controlled_mean_actual_leg_length`, `test_lbar_coordinate_reconstructs_leg_lengths_at_asymmetric_trim`, `test_body_inertia_enforces_left_right_symmetry`, `test_wheel_input_limit_matches_lower_level_current_limit`, `test_height_028_and_beta_zero_reach_requested_endpoints`, `test_mass_and_input_matrices_have_expected_properties`, `test_wheel_height_constraint_is_preserved`, `test_heading_positions_are_relative_to_wheel_midpoint`, `test_heading_velocity_map_contains_path_and_yaw_transport_columns`, `test_heading_velocity_map_reconstructs_world_components`, `test_yaw_column_is_heading_frame_transport_velocity`, `test_absolute_yaw_does_not_change_isotropic_horizontal_mechanics`, `test_requested_beta_point_is_rejected_as_non_equilibrium`, `test_true_static_node_is_controllable_and_lqr_stable` |
| [test_rdob.py](test_rdob.py) | `mechanics`, `make_design`, `test_default_rdob_poles_match_nominal_step_pole`, `test_step_constants_keep_independent_six_channel_layout`, `test_acceleration_elimination_and_input_channel_separation`, `test_only_orthogonal_unmatched_force_is_rejected`, `test_continuous_observer_tracks_all_six_constant_disturbances`, `test_discrete_update_input_timing_and_compensation_sign`, `test_sampled_implementation_converges_to_continuous_solution`, `test_initialization_and_coefficient_rebase_have_no_estimate_jump`, `test_real_robot_trim_design_and_state_order`, `test_invalid_design_is_rejected`, `test_nonfinite_runtime_measurements_are_rejected`, `test_export_cli_and_float32_coefficients`, `test_cli_default_output_and_explicit_override`, `test_exported_cpp_header_compiles_and_matches_python_update` |
| [test_rdob_schedule.py](test_rdob_schedule.py) | `schedule`, `test_direct_schedule_covers_independent_lengths_and_preserves_fixed_bq_zeros`, `test_direct_fit_and_downstream_contract`, `test_piecewise_update_uses_lqr_trim_then_rebases`, `test_piecewise_update_rebases_moving_trim_after_internal_limit`, `test_closed_loop_validation_matrix_matches_explicit_step`, `test_invalid_schedule_points_are_rejected`, `test_invalid_grid_and_fixed_structure_failure`, `test_cli_defaults_to_direct_schedule`, `test_schedule_export_is_deterministic_and_cpp_matches_float32` |
| [test_saturation.py](test_saturation.py) | `test_saturation_uses_perturbation_input_after_absolute_clipping`, `test_batched_simulation_matches_individual_simulations`, `test_nominal_gain_passes_saturated_recovery_and_hits_limits` |
| [test_step_drop.py](test_step_drop.py) | `test_target_lengths_keep_mean_and_match_018m_step`, `test_sudden_018m_drop_hits_lower_contact_but_does_not_recover` |
| [test_symbolic_controllability.py](test_symbolic_controllability.py) | `test_nominal_input_matrix_has_six_independent_columns`, `test_generic_second_order_witness_is_controllable`, `test_second_order_pbh_matrix_shape` |
