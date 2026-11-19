"""One command: additional example images -> calibration -> example results."""
import argparse
from pathlib import Path
from demo_common import ROOT
from calibration_rotation import run_demo as run_rotation
from calibration_static_heading_domain import run_demo as run_static
from example_outputs import generate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'results')
    parser.add_argument('--plots-only',action='store_true',help='Regenerate from existing outputs after strict source/config checks')
    args=parser.parse_args()
    if not args.plots_only:
        for run in [run_rotation,run_static]:
            summary=run(output_dir=args.output_dir)
            if not summary['optimizer_success']:raise RuntimeError('Calibration did not converge; no final example artifacts generated')
    generate(args.output_dir)


if __name__=='__main__':main()
