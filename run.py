import os
import subprocess
import shutil

def main():
	runs=1
	epochs=50
	learning_rate = 0.01
	methods = ['deepall','rsc','mldg','damldg']
	scenarios = ["scn1","scn2","scn3","scn4","scn5","scn6","scn7"]

	for run in range(runs):
		for scenario in scenarios:
			for method in methods:
				weights = os.path.abspath(f"./src/architectures/weights/resnet18_{'imagenet' if scenario=='pacs' else 'voxceleb1'}.h5")
				save_dir = os.path.abspath("./assets/saved_models")
				result_dir = os.path.abspath("./assets/results")
				call_cmd = ["python3", f"./src/main.py", "--epochs", str(epochs), "--scenario", scenario,
					"--learning_rate", str(learning_rate), "--runs", str(run), "--weights", weights, "--validation_freq", str(epochs//2),
					"--result_dir", result_dir, "--save_dir", save_dir, "--method", method
				]
				print(" ".join(call_cmd))
				print("\n")
				subprocess.call(call_cmd)
				# Delete any cache from previous experiments
				if os.path.exists('./logs') and os.path.isdir('./logs'):
					shutil.rmtree('./logs')
				if os.path.exists('./checkpoint') and os.path.isdir('./checkpoint'):
						shutil.rmtree('./checkpoint')

if __name__ == "__main__":
	main()