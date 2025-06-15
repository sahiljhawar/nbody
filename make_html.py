import os

folder = "images"
with open(os.path.join(folder, "index.html"), "w") as f:
    f.write("<html><body><h1>Image Files</h1><ul>\n")
    for filename in os.listdir(folder):
        f.write(f'<li><a href="images/{filename}">{filename}</a></li>\n')
    f.write("</ul></body></html>")
