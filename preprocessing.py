from normalizing.normalizing import run_normalize
from merging_channels.merge import run_merge
from flattening.flattening import run_flattening
from gradient_calculation.gradient_calculation import run_gradient

def main():
    run_normalize()
    run_merge()
    run_flattening()
    run_gradient()
    print("\nPreprocessing complete")

if __name__ == "__main__":
    main()