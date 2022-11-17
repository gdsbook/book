import os

with open('abstracts.md', 'r') as input:
    content = input.read()

chapters = content.split("#")[1:]

for i, chapter in enumerate(chapters):
    i += 1
    fname = f"chapter {i}.md"
    fname = fname.replace(" ","_")
    fname = fname.replace("'","")
    with open(fname, 'w') as output:
        output.write("# "+chapter)
    base = fname.split(".")[0]
    cmd = f"pandoc {fname} -o {base}.docx"
    print(cmd)
    os.system(cmd)

