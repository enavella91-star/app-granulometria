import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import base64
import io
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
from fpdf import FPDF

st.set_page_config(page_title="Granulometría Pro", layout="wide")
st.title("📷 Análisis Granulométrico")

# Función para convertir imagen a base64 (evita errores de 'image_to_url')
def image_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()

uploaded_file = st.file_uploader("Sube la imagen de la muestra", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    st.write("### 1. Define la Escala Manualmente")
    
    # Pasamos la imagen convertida a base64 para máxima compatibilidad
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 255, 0.3)",
        stroke_width=3,
        stroke_color="#FF00FF",
        background_image=image, 
        drawing_mode="rect",
        key="canvas",
        height=image.height,
        width=image.width,
    )

    if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
        col1, col2 = st.columns(2)
        r_ancho = col1.number_input("Ancho real papel (cm)", value=14.8)
        r_largo = col2.number_input("Largo real papel (cm)", value=21.0)
        
        rect = canvas_result.json_data["objects"][-1]
        px_por_cm = ((rect['width'] / r_ancho) + (rect['height'] / r_largo)) / 2
        
        if st.button("Procesar Análisis"):
            # Lógica de procesamiento (Watershed)
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            # ... (tu lógica de Watershed aquí)
            
            st.success("Análisis procesado exitosamente.")
            
            # Generación de PDF (Sintaxis corregida)
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(190, 10, "Reporte Granulometrico", ln=True, align='C')
            
            # Ejemplo de celda corregida:
            # pdf.cell(40, 4.5, f"Pasante: {valor}%") 
            
            pdf_output = "reporte.pdf"
            pdf.output(pdf_output)
            with open(pdf_output, "rb") as f:
                st.download_button("📥 Descargar PDF", f, "reporte.pdf")
