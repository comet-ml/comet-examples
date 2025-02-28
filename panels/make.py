import glob

with open("README.md", "w") as fp:
    for readme_filename in sorted(glob.glob("*/README.md")):
        panel_name, _ = readme_filename.split("/")
        for line in open(readme_filename):
            if line.startswith("####"):
                fp.write(f'\nFor more information, see the panel <a href="https://github.com/comet-ml/comet-examples/blob/master/panels/{panel_name}/README.md">README.md</a>\n')
                break
            fp.write(line)
