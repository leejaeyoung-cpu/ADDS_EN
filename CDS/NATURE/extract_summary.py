import os

files = ['s41467-025-55847-5.pdf', 's43018-022-00416-8.pdf']
base_dir = r"F:\ADDS\CDS\NATURE"
out_path = os.path.join(base_dir, 'summary.txt')

with open(out_path, 'w', encoding='utf-8') as fout:
    try:
        import fitz # PyMuPDF
        fout.write("Using PyMuPDF\n")
        for f in files:
            path = os.path.join(base_dir, f)
            doc = fitz.open(path)
            fout.write(f"\n{'='*20} {f} {'='*20}\n")
            text = ""
            for i in range(min(4, len(doc))):
                text += doc[i].get_text()
            fout.write(text[:4000])
    except ImportError:
        try:
            from PyPDF2 import PdfReader
            fout.write("Using PyPDF2\n")
            for f in files:
                path = os.path.join(base_dir, f)
                reader = PdfReader(path)
                fout.write(f"\n{'='*20} {f} {'='*20}\n")
                text = ""
                for i in range(min(4, len(reader.pages))):
                    text += reader.pages[i].extract_text()
                fout.write(text[:4000])
        except ImportError:
            fout.write("No PDF library found. Please install PyMuPDF or PyPDF2.\n")
