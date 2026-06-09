import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
from fpdf import FPDF
import io

# Configuración de página
st.set_page_config(page_title="Granulometría", layout="wide")
st.title("📷 Análisis Granulométrico")

uploaded_file = st.file_uploader("Sube la imagen de la muestra", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convertimos la imagen cargada a un formato PIL compatible con el canvas
    image = Image.open(uploaded_file).convert("RGB")
    
    st.write("### 1. Define la Escala Manualmente")
    st.write("Dibuja un recuadro fucsia sobre el papel de referencia.")
    
    # El canvas ahora usa el objeto 'image' (PIL) directamente
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
        # Inputs para la escala
        col1, col2 = st.columns(2)
        r_ancho = col1.number_input("Ancho real papel (cm)", value=14.8)
        r_largo = col2.number_input("Largo real papel (cm)", value=21.0)
        
        rect = canvas_result.json_data["objects"][-1]
        px_por_cm = ((rect['width'] / r_ancho) + (rect['height'] / r_largo)) / 2
        
        if st.button("Procesar Análisis Granulométrico"):
            # Lógica Watershed
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl_img = clahe.apply(gray)
            _, thresh = cv2.threshold(cv2.GaussianBlur(cl_img, (7, 7), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            dist = ndimage.distance_transform_edt(thresh)
            local_max = peak_local_max(dist, min_distance=20, labels=thresh)
            mask = np.zeros(dist.shape, dtype=bool)
            mask[tuple(local_max.T)] = True
            markers, _ = ndimage.label(mask)
            labels = watershed(-dist, markers, mask=thresh)

            # Cálculos de áreas
            diametros = []
            for lab in np.unique(labels):
                if lab == 0: continue
                mask_roca = np.where(labels == lab, 255, 0).astype("uint8")
                cnts, _ = cv2.findContours(mask_roca, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts and cv2.contourArea(cnts[0]) > 50:
                    diam_cm = 2 * np.sqrt((cv2.contourArea(cnts[0]) / (px_por_cm**2)) / np.pi)
                    diametros.append(diam_cm * 10)
            
            st.success(f"Procesamiento finalizado. Partículas detectadas: {len(diametros)}")
            # Generación de PDF (sintaxis corregida para evitar errores de cadena)
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(190, 10, "Reporte Granulometrico", ln=True, align='C')
            pdf.output("reporte.pdf")
            with open("reporte.pdf", "rb") as f:
                st.download_button("📥 Descargar PDF", f, "reporte.pdf")
