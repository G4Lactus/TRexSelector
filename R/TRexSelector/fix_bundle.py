with open("bundle_for_cran.sh", "r") as f:
    text = f.read()

text = text.replace('\nrm -rf src/utils/memmap', '')

with open("bundle_for_cran.sh", "w") as f:
    f.write(text)
