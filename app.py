import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import io
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
from scipy.interpolate import interp1d
from fpdf import FPDF

st.set_page_config(page_title="Granulometría", layout="wide")
st.title("📷 Análisis Granulométrico")

mallas = {'600 mm': 600.0, '500 mm': 500.0, '400 mm': 400.0, '300 mm': 300.0, '200 mm': 200.0, '150 mm': 150.0,
          '5"': 127.0, '4"': 101.6, '3"': 76.2, '2"': 50.8, '1.5"': 38.1, '1"': 25.4, '3/4"': 19.05,
          '1/2"': 12.7, '3/8"': 9.51, '1/4"': 6.35, '#4': 4.76, '#8': 2.38, '#14': 1.41, 
          '#28': 0.595, '#48': 0.297, '#100': 0.149, '#200': 0.074}

uploaded_file = st.file_uploader("Sube la imagen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convertir bytes a imagen PIL correctamente
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    
    st.write("### 1. Define la Escala Manualmente")
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 255, 0.3)", stroke_width=3, stroke_color="#FF00FF",
        background_image=image, drawing_mode="rect", key="canvas",
        height=image.height, width=image.width,
    )

    if canvas_result.json_data and len(canvas_result.json_data["objects"]) > 0:
        r_ancho = st.number_input("Ancho real papel (cm)", value=14.8)
        r_largo = st.number_input("Largo real papel (cm)", value=21.0)
        
        rect = canvas_result.json_data["objects"][-1]
        px_por_cm = ((rect['width'] / r_ancho) + (rect['height'] / r_largo)) / 2
        
        if st.button("Procesar Análisis"):
            # Lógica de procesamiento (Watershed simplificado)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl_img = clahe.apply(gray)
            _, thresh = cv2.threshold(cv2.GaussianBlur(cl_img, (7, 7), 0), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            dist = ndimage.distance_transform_edt(thresh)
            local_max = peak_local_max(dist, min_distance=20, labels=thresh)
            mask = np.zeros(dist.shape, dtype=bool); mask[tuple(local_max.T)] = True
            markers, _ = ndimage.label(mask)
            labels = watershed(-dist, markers, mask=thresh)

            # Cálculos
            diametros, masas = [], []
            for lab in np.unique(labels):
                if lab == 0: continue
                mask_roca = np.where(labels == lab, 255, 0).astype("uint8")
                cnts, _ = cv2.findContours(mask_roca, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts and cv2.contourArea(cnts[0]) > 50:
                    diam_cm = 2 * np.sqrt((cv2.contourArea(cnts[0]) / (px_por_cm**2)) / np.pi)
                    diametros.append(diam_cm * 10)
                    masas.append(((4/3) * np.pi * ((diam_cm/2)**3)) * 2.65)
            
            df = pd.DataFrame({'D': diametros, 'M': masas})
            total = df['M'].sum()
            res = []
            acum = 0
            for k, v in mallas.items():
                m_ret = df[df['D'] >= v]['M'].sum()
                porc = (m_ret / total) * 100
                res.append({'Malla': k, 'Pasante': 100 - porc})
            
            df_final = pd.DataFrame(res)
            
            # Generación de PDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(190, 10, "Resultados Granulometria", ln=True, align='C')
            pdf.set_font("Arial", '', 8)
            for _, row in df_final.iterrows():
                pdf.cell(60, 6, str(row['Malla']), 1)
                pdf.cell(60, 6, f"{row['Pasante']:.2f}%", 1)
                pdf.ln()
            
            pdf_bytes = pdf.output(dest='S').encode('latin-1')
            st.download_button("📥 Descargar PDF", pdf_bytes, "reporte.pdf")
