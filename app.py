
### BOILDER PLATE PDF FILE LOADER/VIEWER


import streamlit as st

file = st.file_uploader("Upload a PDF file", type="pdf")


import PyPDF2

if file is not None:
    # Read the PDF file
    pdf_reader = PyPDF2.PdfFileReader(file)
    # Extract the content
    content = ""
    for page in range(pdf_reader.getNumPages()):
        content += pdf_reader.getPage(page).extractText()
    # Display the content
    st.write(content)