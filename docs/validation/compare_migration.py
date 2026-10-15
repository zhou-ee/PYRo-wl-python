"""Read-only, fresh-process comparison of legacy and migrated implementations.

Worker usage: python compare_migration.py old|new ROOT MASS OUTPUT_DIRECTORY
Each process imports exactly one repository and emits numerical evidence.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass, replace
from pathlib import Path
import hashlib
import importlib
import json
import re
import sys

import numpy as np


def main():
    sys.dont_write_bytecode = True
    side, root_arg, mass_arg, output_arg = sys.argv[1:]
    root, output, mass = Path(root_arg), Path(output_arg), float(mass_arg)
    sys.path.insert(0, str(root / 'src'))
    sys.path.insert(1, str(root))
    output.mkdir(parents=True, exist_ok=True)
    arrays, notes = {}, {}

    def module(path):
        return importlib.import_module('wheel_leg_lqr.' + path)

    if side == 'old':
        cfg, air_cfg = module('core.config'), module('airborne.config')
        # Only process-local snapshots change; legacy repository files stay intact.
        cfg.PARAMETERS = replace(cfg.PARAMETERS, body_mass=mass)
        air_cfg.PARAMETERS = replace(air_cfg.PARAMETERS, body_mass=mass)
        cfg.TRIM_BETA_1 = cfg.TRIM_BETA_2 = cfg.compute_equilibrium_beta(cfg.TRIM_HEIGHT)
        cfg.TRIM_L_BAR = float(cfg.TRIM_HEIGHT / np.cos(cfg.TRIM_BETA_1))
        ground = module('scheduling.leg_length_schedule')
        air = module('airborne.schedule')
        rdob = module('scheduling.rdob_schedule')
        node_module = module('rdob')
        contact = drop = module('analysis.step_drop')
        saturation = module('control.saturation')
        air_model = module('airborne.model')
        beta = cfg.compute_equilibrium_beta_from_length
        gas = cfg.compute_gas_spring_force
        make_ground = ground.compute_gain_node
        make_rdob = node_module.compute_rdob_node
        make_contact = contact._mass_and_input
        run_drop = drop.simulate_step_drop
        pitch = air_model.compute_leg_force_pitch_effect
        from data_process.com_fit import fit_leg_com_data
        cad = fit_leg_com_data(root / 'data_process' / '清洗后重心.xlsx')
        render_ground, render_air, render_rdob = (
            ground.render_header, air.render_airborne_header, rdob.render_rdob_schedule_header)
    else:
        physical = module('config')
        robot = replace(physical.DEFAULT_ROBOT, body_mass=mass)
        cfg, air_cfg = module('models.ground.config'), module('models.airborne.config')
        ground = module('scheduling.ground_lqr')
        air = module('scheduling.airborne_lqr')
        rdob = module('scheduling.rdob')
        node_module = module('scheduling.rdob_node')
        contact, drop = module('models.contact.model'), module('simulation.step_drop')
        saturation = module('simulation.saturation')
        air_model = module('models.airborne.model')
        trim, geometry = module('models.ground.trim'), module('models.leg_geometry')
        beta = lambda length: trim.compute_equilibrium_beta_from_length(length, robot)
        gas = lambda length: geometry.compute_gas_spring_force(length, robot)
        make_ground = lambda *args: ground.compute_gain_node(*args, parameters=robot)
        make_rdob = lambda *args: node_module.compute_rdob_node(*args, parameters=robot)
        make_contact = lambda q: contact._mass_and_input(q, robot)
        run_drop = lambda **kw: drop.simulate_step_drop(parameters=robot, **kw)
        pitch = lambda *args, **kw: air_model.compute_leg_force_pitch_effect(*args, parameters=robot, **kw)
        cad = module('calibration.leg_com').fit_leg_com_data(
            root / 'data/reference' / '清洗后重心.xlsx', parameters=robot)
        render_ground = module('artifacts.ground_lqr').render_header
        render_air = module('artifacts.airborne_lqr').render_airborne_header
        render_rdob = module('artifacts.rdob').render_rdob_schedule_header

    def collect(prefix, value):
        if is_dataclass(value):
            for field in fields(value):
                collect(prefix + '/' + field.name, getattr(value, field.name))
        elif isinstance(value, dict):
            for key, part in value.items():
                collect(prefix + '/' + str(key), part)
        elif isinstance(value, np.ndarray):
            arrays[prefix] = value
        elif isinstance(value, (list, tuple)):
            if not value:
                arrays[prefix] = np.array([])
            elif all(isinstance(x, (float, int, bool, np.number)) for x in value):
                arrays[prefix] = np.asarray(value)
            else:
                for index, part in enumerate(value):
                    collect(prefix + '/' + str(index), part)
        elif isinstance(value, (float, int, bool, np.number)):
            arrays[prefix] = np.asarray(value)
        elif value is None or isinstance(value, str):
            notes[prefix] = value
        else:
            raise TypeError((prefix, type(value)))

    def header(prefix, content):
        (output / (prefix + '.h')).write_text(content, encoding='utf-8')
        pattern = r'\b(?:float|uint32_t)\s+(\w+)\s*(?:\[[^;=]*?)?\s*=\s*([^;]+);'
        number = r'(?<![A-Za-z_])[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?'
        for name, expression in re.findall(pattern, content, flags=re.S):
            arrays['header/' + prefix + '/' + name] = np.array(
                [float(x) for x in re.findall(number, expression)])

    settings = {
        'mass': mass, 'ground_q': cfg.Q_DIAG.tolist(), 'ground_r': cfg.R_DIAG.tolist(),
        'air_q': air_cfg.Q_DIAG.tolist(), 'air_r': air_cfg.R_DIAG.tolist(),
        'ground_sample_time': cfg.SAMPLE_TIME, 'air_sample_time': air_cfg.SAMPLE_TIME,
        'rdob_bandwidth': list(node_module.DEFAULT_BANDWIDTH_HZ),
    }
    hardware = cfg.PARAMETERS if side == 'old' else robot
    for name in ('wheel_mass', 'wheel_rolling_inertia', 'wheel_radius', 'leg_mass',
                 'body_mass', 'hip_half_track', 'wheel_half_track', 'body_com_from_hip',
                 'body_inertia', 'gravity', 'min_leg_length', 'max_leg_length'):
        arrays['settings/' + name] = np.asarray(getattr(hardware, name))
    arrays['settings/ground_q'] = cfg.Q_DIAG
    arrays['settings/ground_r'] = cfg.R_DIAG
    arrays['settings/air_q'] = air_cfg.Q_DIAG
    arrays['settings/air_r'] = air_cfg.R_DIAG
    lengths = np.linspace(0.18, 0.38, 201)
    arrays['trim/beta'] = np.array([beta(length) for length in lengths])
    arrays['trim/gas'] = np.array([gas(length) for length in lengths])
    if side == 'old':
        arrays['beta_export/coefficients'] = np.asarray(cfg.LEG_COM_BETA_EQ_POLY)
    else:
        arrays['beta_export/coefficients'] = trim.fit_equilibrium_beta(robot)
    arrays['beta_export/errors'] = np.polynomial.polynomial.polyval(
        lengths, arrays['beta_export/coefficients']) - arrays['trim/beta']

    print(f'{side} {mass:g}: ground schedule', flush=True)
    gfit = (ground.compute_gain_schedule(step=0.02) if side == 'old' else
            ground.compute_gain_schedule(step=0.02, parameters=robot))
    for key in ('l1_grid', 'l2_grid', 'gain_coefficients', 'input_coefficients', 'fitted_gains',
                'max_abs_gain_error', 'rms_gain_error', 'max_raw_spectral_radius',
                'max_fitted_spectral_radius', 'saturation_validation_node_count', 'saturation_validation'):
        collect('ground_schedule/' + key, getattr(gfit, key))
    for key in ('mass', 'A', 'B', 'stiffness', 'input_matrix', 'equilibrium_input', 'equilibrium_residual'):
        arrays['ground_nodes/' + key] = np.stack([getattr(n.linearization, key) for n in gfit.nodes])
    for key in ('Ad', 'Bd', 'gain', 'closed_loop_poles'):
        arrays['ground_nodes/' + key] = np.stack([getattr(n, key) for n in gfit.nodes])
    arrays['ground_nodes/q'] = np.stack([n.linearization.geometry.q for n in gfit.nodes])
    for suffix, delta in (('trim', np.zeros(9)), ('perturbed', np.array([.1, .2, .03, -.02, .005, .04, -.03, 0., .015]))):
        values = [make_contact(np.r_[n.linearization.geometry.q,
                                    n.linearization.geometry.kappa_g, 0.] + delta)[:3] for n in gfit.nodes]
        for index, key in enumerate(('mass', 'gravity', 'input_matrix')):
            arrays[f'contact/{suffix}/{key}'] = np.stack([value[index] for value in values])
    header('ground', render_ground(gfit))

    print(f'{side} {mass:g}: airborne schedule', flush=True)
    afit = (air.compute_airborne_schedule(step=0.01) if side == 'old' else
            air.compute_airborne_schedule(step=0.01, parameters=robot))
    for key in ('l1_grid', 'l2_grid', 'sample_time', 'q_diag', 'r_diag', 'gain_coefficients',
                'fitted_gains', 'max_abs_gain_error', 'max_relative_gain_error',
                'max_raw_shape_radius', 'max_float32_shape_radius', 'validation_node_count'):
        collect('air_schedule/' + key, getattr(afit, key))
    for key in ('mass', 'A', 'B', 'Q_shape', 'state_projection', 'state_lift', 'A_shape', 'B_shape'):
        arrays['air_nodes/' + key] = np.stack([getattr(n.linearization, key) for n in afit.nodes])
    for key in ('Ad', 'Bd', 'Ad_shape', 'Bd_shape', 'gain', 'shape_poles'):
        arrays['air_nodes/' + key] = np.stack([getattr(n, key) for n in afit.nodes])
    header('airborne', render_air(afit))

    print(f'{side} {mass:g}: RDOB schedule', flush=True)
    rfit = (rdob.compute_rdob_schedule(step=0.02) if side == 'old' else
            rdob.compute_rdob_schedule(step=0.02, parameters=robot))
    for key in ('lower', 'upper', 'rates', 'sample_time', 'damping', 'training_points',
                'holdout_points', 'diagnostics'):
        collect('rdob_schedule/' + key, getattr(rfit, key))
    for name, fit in rfit.fits.items():
        collect('rdob_schedule/fits/' + name, fit)
    points = np.vstack((rfit.training_points, rfit.holdout_points))
    rnodes = [make_rdob(*point).observer for point in points]
    for key in ('gamma', 'position_gain', 'velocity_gain', 'step_poles', 'step_position_gain',
                'step_velocity_gain', 'step_input_gain'):
        arrays['rdob_nodes/' + key] = np.stack([getattr(n, key) for n in rnodes])
    constants = [module('control.rdob').trapezoidal_step_constants(n.rates, n.sample_time) for n in rnodes]
    for index, key in enumerate(('step_poles', 'step_endpoint_weight', 'step_input_gain', 'step_velocity_scale')):
        arrays['rdob_constants/' + key] = np.stack([part[index] for part in constants])
    header('rdob', render_rdob(rfit))

    print(f'{side} {mass:g}: trajectories and analysis', flush=True)
    for index, pair in enumerate(((.28, .28), (.24, .32), (.18, .38))):
        node = make_ground(*pair)
        error = np.zeros(14); error[[0, 4, 10, 12]] = [.02, .03, .25, -.25]
        limits = {} if side == 'old' else dict(
            lower_limits=robot.input_lower_limits, upper_limits=robot.input_upper_limits)
        trajectory = saturation.simulate_saturated_closed_loop(
            node.Ad, node.Bd, node.gain, node.linearization.equilibrium_input, error,
            horizon_s=0.5, sample_time=cfg.SAMPLE_TIME, **limits)
        collect(f'saturation/{index}', trajectory)
        for com in (0., .03):
            collect(f'pitch/{index}/{com}', pitch(*pair, body_com_x_from_hip=com))
    for height, leg in ((.18, 0), (.08, 1)):
        collect(f'step_drop/{height}/{leg}', run_drop(
            step_height=height, down_leg=leg, mean_leg_length=.28, horizon_s=.2))
    # Analysis functions use module default hardware; align only in this worker.
    if side == 'new':
        for path in ('analysis.bandwidth', 'analysis.qr_optimizer'):
            mod = module(path)
            if hasattr(mod, 'DEFAULT_ROBOT'):
                mod.DEFAULT_ROBOT = robot
            mod.compute_equilibrium_beta = lambda height: trim.compute_equilibrium_beta(height, robot)
            original_linearize = module('models.ground.model').linearize_static_trim
            mod.linearize_static_trim = lambda *a, **kw: original_linearize(*a, parameters=robot, **kw)
    collect('bandwidth', module('analysis.bandwidth').analyze_controller_bandwidth(h_target=.24))
    qr = module('analysis.qr_optimizer')
    n = make_ground(.24, .24)
    e0 = np.zeros(14); e0[[0, 4, 10, 12]] = [.02, .03, .25, -.25]
    collect('qr_candidate', qr.evaluate_qr_candidate(
        cfg.Q_DIAG, cfg.R_DIAG, n.Ad, n.Bd, e0, .24, beta(.24)))
    collect('qr_optimizer_changed', qr.optimize_qr_matrix(max_iter=1))
    collect('cad', cad)
    workbook = root / ('data_process/清洗后重心.xlsx' if side == 'old' else 'data/reference/清洗后重心.xlsx')
    settings['workbook_sha256'] = hashlib.sha256(workbook.read_bytes()).hexdigest()
    settings['imported_package'] = str(importlib.import_module('wheel_leg_lqr').__file__)
    settings['array_count'] = len(arrays)
    np.savez_compressed(output / 'arrays.npz', **arrays)
    (output / 'manifest.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    (output / 'notes.json').write_text(json.dumps(notes, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'{side} {mass:g}: saved {len(arrays)} arrays', flush=True)


if __name__ == '__main__':
    main()
