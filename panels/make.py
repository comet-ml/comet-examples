import glob

with open("README.md", "w") as fp:
    for readme_filename in glob.glob("*/README.md"):
        print(readme_filename)
        for line in open(readme_filename):
            if line.startswith("####"):
                break
            fp.write(line)
