from PyPDF2 import PdfReader, PdfWriter

def split_pdf(file_path, pages_per_part=200):
    reader = PdfReader(file_path)
    total_pages = len(reader.pages)
    part_num = 1

    for i in range(0, total_pages, pages_per_part):
        writer = PdfWriter()
        for j in range(i, min(i + pages_per_part, total_pages)):
            writer.add_page(reader.pages[j])

        output_path = f"part{part_num}.pdf"
        with open(output_path, "wb") as f:
            writer.write(f)
        print(f"✅ Created {output_path}")
        part_num += 1

split_pdf("Medical_book.pdf", pages_per_part=200)
