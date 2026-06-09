import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
from scipy.interpolate import interp1d
from fpdf import FPDF
import tempfile
import os
from datetime import datetime

# ==========================================
# CONFIGURACIÓN
# ==========================================
st.set_page_config(page_title="Análisis Granulométrico", layout="centered")
st.title("📷 Granulometría por Visión Artificial")

DENSIDAD_SOLIDO = 2.65

# Diccionario ampliado desde 600 mm hasta #200
mallas = {
    '600 mm': 600.0, '500 mm': 500.0, '400 mm': 400.0, '300 mm': 300.0, '200 mm': 200.0, '150 mm': 150.0,
    '5"': 127.0, '4"': 101.6, '3"': 76.2, '2"': 50.8, '1.5"': 38.1, '1"': 25.4, '3/4"': 19.05,
    '1/2"': 12.7, '3/8"': 9.51, '1/4"': 6.35, '#4': 4.76, '#8': 2.38,
    '#14': 1.41, '#28': 0.595, '#48': 0.297, '#100': 0.149, '#200': 0.074
}

# ==========================================
# DATOS DE LA MUESTRA (INTERFAZ)
# ==========================================
st.markdown("### Datos de la Muestra")
col_fecha, col_lugar = st.columns(2)
with col_fecha:
    fecha_input = st.date_input("Fecha del análisis")
with col_lugar:
    lugar_input = st.text_input("Lugar de donde viene la muestra", placeholder="Ej: Cantera Sur, Nivel 4")

st.markdown("### Configuración de Referencia")
col1, col2 = st.columns(2)
with col1:
    ANCHO_PAPEL_CM = st.number_input("Ancho del papel ref. (cm)", min_value=0.1, value=14.8, step=0.1)
with col2:
    LARGO_PAPEL_CM = st.number_input("Largo del papel ref. (cm)", min_value=0.1, value=21.0, step=0.1)

uploaded_file = st.file_uploader("Sube la imagen de las rocas", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Leer imagen
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    st.image(img_rgb, caption="Imagen original", use_column_width=True)
    
    if not lugar_input:
        st.warning("⚠️ Por favor, ingresa el lugar de la muestra antes de procesar.")
        
    elif st.button("Procesar y Generar Reporte"):
        with st.spinner('Analizando imagen y calculando segmentación...'):
            
            # 2. Detectar Escala
            _, thresh_papel = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
            contornos_papel, _ = cv2.findContours(thresh_papel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contornos_papel:
                st.error("No se detectó el papel. Asegúrate de que esté bien iluminado.")
                st.stop()
                
            contorno_mayor = max(contornos_papel, key=cv2.contourArea)
            rect = cv2.minAreaRect(contorno_mayor)
            largo_px = max(rect[1])
            px_por_cm = largo_px / LARGO_PAPEL_CM
            
            # 3. CLAHE + Watershed
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl_img = clahe.apply(img_gray)
            blur = cv2.GaussianBlur(cl_img, (7, 7), 0)
            
            _, thresh_rocas = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            dist_transform = ndimage.distance_transform_edt(thresh_rocas)
            local_max = peak_local_max(dist_transform, min_distance=20, labels=thresh_rocas)
            mask = np.zeros(dist_transform.shape, dtype=bool)
            mask[tuple(local_max.T)] = True
            marcadores, _ = ndimage.label(mask)
            labels = watershed(-dist_transform, marcadores, mask=thresh_rocas)

            # 4. Cálculo
            diametros_mm, masas_g = [], []
            for label in np.unique(labels):
                if label == 0: continue
                mask_roca = np.zeros(img_gray.shape, dtype="uint8")
                mask_roca[labels == label] = 255
                cnts, _ = cv2.findContours(mask_roca, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(cnts) == 0: continue
                c = cnts[0]
                area_px = cv2.contourArea(c)
                if area_px < 50: continue
                
                area_cm2 = area_px / (px_por_cm ** 2)
                diametro_cm = 2 * np.sqrt(area_cm2 / np.pi)
                diametros_mm.append(diametro_cm * 10)
                volumen_cm3 = (4/3) * np.pi * ((diametro_cm / 2) ** 3)
                masas_g.append(volumen_cm3 * DENSIDAD_SOLIDO)

            df_particulas = pd.DataFrame({'Diametro_mm': diametros_mm, 'Masa_g': masas_g})
            if df_particulas.empty:
                st.error("No se detectaron rocas.")
                st.stop()
                
            # 5. Distribución
            masa_total = df_particulas['Masa_g'].sum()
            resultados_mallas, masa_acumulada = [], 0
            aberturas = list(mallas.values())
            nombres = list(mallas.keys())

            for i in range(len(aberturas)):
                if i == 0:
                    masa_retenida = df_particulas[df_particulas['Diametro_mm'] >= aberturas[i]]['Masa_g'].sum()
                else:
                    masa_retenida = df_particulas[(df_particulas['Diametro_mm'] < aberturas[i-1]) & 
                                                  (df_particulas['Diametro_mm'] >= aberturas[i])]['Masa_g'].sum()
                
                porcentaje_retenido = (masa_retenida / masa_total) * 100
                masa_acumulada += porcentaje_retenido
                resultados_mallas.append({
                    'Malla': nombres[i], 'Abertura_mm': aberturas[i], 
                    '% Retenido': porcentaje_retenido, '% Pasante': 100 - masa_acumulada
                })

            masa_finos = df_particulas[df_particulas['Diametro_mm'] < aberturas[-1]]['Masa_g'].sum()
            resultados_mallas.append({
                'Malla': 'Finos', 'Abertura_mm': 0,
                '% Retenido': (masa_finos / masa_total) * 100, '% Pasante': 0
            })
            
            df_dist = pd.DataFrame(resultados_mallas)
            f_interp = interp1d(df_dist['% Pasante'], df_dist['Abertura_mm'], fill_value="extrapolate")
            p80 = float(f_interp(80.0))
            p80_inch = p80 / 25.4 

            # 6. Gráfico
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(df_dist['Abertura_mm'], df_dist['% Pasante'], 'o-', label='Curva Granulométrica')
            ax.axhline(y=80, color='r', linestyle='--', label='80% Pasante')
            ax.axvline(x=p80, color='g', linestyle='--', label=f'P80 = {p80:.2f} mm')
            ax.set_xscale('log')
            ax.set_xlabel('Abertura de Malla (mm)')
            ax.set_ylabel('% Pasante Acumulado')
            ax.set_title('Distribución Granulométrica')
            ax.grid(True, which="both", ls="-")
            ax.legend()
            
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown(f"<h2 style='text-align: center; color: #155724; background-color: #d4edda; padding: 15px; border-radius: 10px;'>"
                        f"P80 Calculado:<br>{p80:.2f} mm &nbsp;|&nbsp; {p80_inch:.2f} pulgadas</h2>", 
                        unsafe_allow_html=True)
            st.pyplot(fig) 
            
            # ==========================================
            # 7. GENERACIÓN DEL PDF AJUSTADO (MÁS FILAS)
            # ==========================================
            with tempfile.TemporaryDirectory() as tmpdirname:
                img_path = os.path.join(tmpdirname, "foto.jpg")
                plot_path = os.path.join(tmpdirname, "plot.png")
                pdf_path = os.path.join(tmpdirname, "reporte.pdf")
                
                cv2.imwrite(img_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
                fig.savefig(plot_path, bbox_inches='tight')
                
                pdf = FPDF(orientation='P', unit='mm', format='A4')
                pdf.add_page()
                pdf.set_margins(10, 10, 10)
                
                pdf.set_font("Arial", 'B', 15)
                pdf.cell(190, 8, txt="Reporte de Análisis Granulométrico", ln=True, align='C')
                
                pdf.set_font("Arial", '', 11)
                fecha_formateada = fecha_input.strftime("%d/%m/%Y")
                pdf.cell(190, 6, txt=f"Fecha: {fecha_formateada}   |   Lugar: {lugar_input}", ln=True, align='C')
                
                pdf.set_font("Arial", 'B', 13)
                pdf.set_text_color(0, 100, 0)
                pdf.cell(190, 8, txt=f"P80: {p80:.2f} mm  ({p80_inch:.2f} pulgadas)", ln=True, align='C')
                pdf.set_text_color(0, 0, 0) 
                
                y_imagenes = pdf.get_y() + 2
                
                h_img, w_img = img_rgb.shape[:2]
                aspect_ratio_img = w_img / h_img
                
                # Se reduce ligeramente el máximo a 70mm para dejar espacio a la tabla más larga
                max_h_permitido = 70 
                max_w_permitido = 90
                
                if (max_w_permitido / aspect_ratio_img) <= max_h_permitido:
                    img_pdf_w = max_w_permitido
                    img_pdf_h = max_w_permitido / aspect_ratio_img
                else:
                    img_pdf_h = max_h_permitido
                    img_pdf_w = max_h_permitido * aspect_ratio_img

                x_img = 10 + ((90 - img_pdf_w) / 2)
                pdf.image(img_path, x=x_img, y=y_imagenes, w=img_pdf_w, h=img_pdf_h)
                
                graph_w = 95
                graph_h = graph_w / 1.6 
                pdf.image(plot_path, x=105, y=y_imagenes, w=graph_w, h=graph_h)
                
                y_max_imagenes = y_imagenes + max(img_pdf_h, graph_h)
                pdf.set_y(y_max_imagenes + 5)
                
                # --- TABLA COMPACTADA ---
                pdf.set_font("Arial", 'B', 8) # Letra más pequeña para acomodar 24 filas
                margen_izq = 45 
                
                pdf.set_x(margen_izq)
                pdf.cell(40, 5, 'Malla', 1, 0, 'C')
                pdf.cell(40, 5, 'Abertura (mm)', 1, 0, 'C')
                pdf.cell(40, 5, '% Pasante', 1, 1, 'C') 
                
                pdf.set_font("Arial", '', 8)
                for index, row in df_dist.iterrows():
                    pdf.set_x(margen_izq)
                    # Altura de celda a 4.5mm para evitar que salte de hoja
                    pdf.cell(40, 4.5, str(row['Malla']), 1, 0, 'C')
                    pdf.cell(40, 4.5, f"{row['Abertura_mm']:.2f}", 1, 0, 'C')
                    pdf.cell(40, 4.5, f"{row['% Pas
