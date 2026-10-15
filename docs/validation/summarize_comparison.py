"""Compare worker evidence and fail on unexplained numerical differences."""
from pathlib import Path
import json
import sys
import numpy as np


def main():
    output = Path(sys.argv[1])
    report = {'rtol': 1e-10, 'atol': 1e-11, 'profiles': {}}
    for mass in ('12', '18.9255'):
        old_dir, new_dir = output / ('old_' + mass), output / ('new_' + mass)
        old = np.load(old_dir / 'arrays.npz')
        new = np.load(new_dir / 'arrays.npz')
        old_manifest = json.loads((old_dir / 'manifest.json').read_text(encoding='utf-8'))
        new_manifest = json.loads((new_dir / 'manifest.json').read_text(encoding='utf-8'))
        assert old_manifest['workbook_sha256'] == new_manifest['workbook_sha256']
        assert {k:v for k,v in old_manifest.items() if k not in ('imported_package', 'array_count')} == {
            k:v for k,v in new_manifest.items() if k not in ('imported_package', 'array_count')}
        missing = sorted(set(old.files) ^ set(new.files))
        rows, groups = [], {}
        for key in sorted(set(old.files) & set(new.files)):
            a, b = old[key], new[key]
            shape_equal = a.shape == b.shape
            exact = shape_equal and np.array_equal(a, b)
            finite = np.all(np.isfinite(a)) and np.all(np.isfinite(b))
            close = bool(shape_equal and finite and np.allclose(a, b, rtol=1e-10, atol=1e-11))
            delta = float(np.max(np.abs(a.astype(complex) - b.astype(complex)))) if shape_equal and a.size else 0.
            category = 'same' if exact else 'roundoff' if close else 'unexpected'
            explanation = None
            if not close:
                if key.startswith('beta_export/') or key == 'header/ground/BETA_TRIM_POLY_COEF':
                    category, explanation = 'intentional', 'Equilibrium-angle export is now refitted from the active physical parameters.'
                elif key in ('qr_optimizer_changed/0', 'qr_optimizer_changed/3/q_diag') and (
                        a.shape == b.shape == (14,) and np.array_equal(np.delete(a,6), np.delete(b,6)) and a[6] == 800. and b[6] == 600.):
                    category, explanation = 'intentional', 'Unoptimized yaw/roll Q now comes from config; legacy optimizer hardcoded roll Q=800 while config uses 600.'
                elif mass == '18.9255' and (key in ('cad/beta_eq', 'cad/poly_beta_eq') or key.startswith('cad/metrics/') and 'beta_eq' in key):
                    category, explanation = 'intentional', 'Legacy CAD fitter hardcodes 12 kg; new fitter reads the requested 18.9255 kg.'
            row = dict(key=key, shape=list(a.shape), category=category, max_abs_difference=delta)
            if explanation:
                row['explanation'] = explanation
            rows.append(row)
            group = groups.setdefault(key.split('/')[0], {'same':0, 'roundoff':0, 'intentional':0, 'unexpected':0, 'max_abs_difference':0.})
            group[category] += 1
            group['max_abs_difference'] = max(group['max_abs_difference'], delta)
        old_notes = json.loads((old_dir / 'notes.json').read_text(encoding='utf-8'))
        new_notes = json.loads((new_dir / 'notes.json').read_text(encoding='utf-8'))
        note_changes = {k: [old_notes.get(k), new_notes.get(k)] for k in old_notes.keys() | new_notes.keys() if old_notes.get(k) != new_notes.get(k)}
        stats = {category:sum(row['category'] == category for row in rows) for category in ('same', 'roundoff', 'intentional', 'unexpected')}
        profile = dict(count=len(rows), missing_keys=missing, counts=stats, groups=groups,
                       note_changes=note_changes, differences=[row for row in rows if row['category'] != 'same'])
        profile['beta_export_old_max_error_deg'] = float(np.degrees(np.max(np.abs(old['beta_export/errors']))))
        profile['beta_export_new_max_error_deg'] = float(np.degrees(np.max(np.abs(new['beta_export/errors']))))
        profile['core_max_abs_difference'] = max((row['max_abs_difference'] for row in rows if row['category'] in ('same','roundoff') and not row['key'].startswith('qr_optimizer_changed/')), default=0.)
        expected_beta = np.arctan(new['settings/leg_mass'] * new['cad/x_com'] / (
            float(mass) * new['cad/L'] / 2 + new['settings/leg_mass'] * new['cad/L_wp']))
        profile['new_cad_beta_formula_max_error'] = float(np.max(np.abs(expected_beta - new['cad/beta_eq'])))
        assert profile['new_cad_beta_formula_max_error'] == 0.
        assert profile['beta_export_new_max_error_deg'] < 0.02
        profile['passed'] = not missing and not note_changes and stats['unexpected'] == 0
        report['profiles'][mass] = profile
        print(mass, 'kg:', json.dumps(stats), 'core_max_abs=', profile['core_max_abs_difference'], 'passed=', profile['passed'])
        for row in rows:
            if row['category'] == 'unexpected':
                print('UNEXPECTED', row)
    old = np.load(output / 'old_12/arrays.npz')
    new = np.load(output / 'new_18.9255/arrays.npz')
    report['unaligned_defaults'] = {
        key: float(np.max(np.abs(old[key] - new[key]))) for key in
        ('trim/beta', 'ground_nodes/mass', 'ground_nodes/gain', 'air_nodes/gain', 'rdob_nodes/gamma')}
    report['passed'] = all(profile['passed'] for profile in report['profiles'].values())
    (output / 'comparison.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
