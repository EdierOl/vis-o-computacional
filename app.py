import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import psycopg2
from psycopg2 import sql
import json
import os

# Configuração da página Streamlit
st.set_page_config(page_title="Segmentação de Imagens", layout="centered")

# --- CONFIGURAÇÃO DO BANCO DE DADOS (NEON.TECH) ---
# Substitua a string abaixo pela connection string fornecida pelo Neon.tech
# Exemplo: "postgres://usuario:senha@ep-nome-do-host.us-east-2.aws.neon.tech/nome_do_banco"
NEON_DB_URL = os.getenv("DATABASE_URL", "postgresql://neondb_owner:npg_8zU9QbYsTedf@ep-late-dream-ayxdsbkh-pooler.c-5.us-east-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require")

def init_db():
    """Cria a tabela no banco de dados caso ela não exista."""
    try:
        conn = psycopg2.connect(NEON_DB_URL)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS image_logs (
                id SERIAL PRIMARY KEY,
                filename VARCHAR(255),
                total_objects INT,
                characteristics JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        st.error(f"Erro ao conectar ou inicializar o banco: {e}")

def save_to_db(filename, total_objects, characteristics_dict):
    """Salva os dados extraídos da imagem no Neon.tech"""
    try:
        conn = psycopg2.connect(NEON_DB_URL)
        cursor = conn.cursor()
        
        insert_query = """
            INSERT INTO image_logs (filename, total_objects, characteristics)
            VALUES (%s, %s, %s)
        """
        cursor.execute(insert_query, (filename, total_objects, json.dumps(characteristics_dict)))
        
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Erro ao salvar no banco: {e}")
        return False

# --- CARREGAMENTO DO MODELO CV ---
@st.cache_resource
def load_model():
    # O YOLO baixa os pesos (yolov8n-seg.pt) automaticamente na primeira vez
    return YOLO('yolov8n-seg.pt')

# --- INTERFACE E PIPELINE ---
def main():
    st.title("Sistema de Segmentação de Imagens")
    st.write("Faça o upload da imagem, processe e envie os dados extraídos para o Neon.tech.")
    
    # Tenta inicializar a tabela no banco
    init_db()
    
    model = load_model()
    
    uploaded_file = st.file_uploader("Escolha uma imagem...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Mostra a imagem original
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagem Original", use_column_width=True)
        
        # Botão para processar
        if st.button("Processar Imagem e Analisar Dados"):
            with st.spinner("Realizando segmentação e análise..."):
                # Converter PIL Image para Numpy Array / BGR para OpenCV
                img_array = np.array(image)
                if img_array.shape[-1] == 4: # Remove canal alpha se for PNG
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                else:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                # Inferência do YOLO
                results = model(img_array)
                result = results[0]
                
                # Extração de Características
                boxes = result.boxes
                class_names = result.names
                
                characteristics = {}
                total_objects = len(boxes)
                
                for box in boxes:
                    cls_id = int(box.cls[0])
                    cls_name = class_names[cls_id]
                    
                    if cls_name in characteristics:
                        characteristics[cls_name] += 1
                    else:
                        characteristics[cls_name] = 1
                
                # Renderizar imagem com as máscaras/boxes aplicadas
                res_plotted = result.plot()
                # Converter de volta para RGB para mostrar no Streamlit
                res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                
                # Exibir resultados na UI
                st.subheader("Resultado da Segmentação")
                st.image(res_plotted_rgb, caption="Imagem Processada", use_column_width=True)
                
                st.subheader("Características Extraídas")
                st.write(f"**Total de objetos detectados:** {total_objects}")
                st.json(characteristics)
                
                # Salvar no banco
                if save_to_db(uploaded_file.name, total_objects, characteristics):
                    st.success("✅ Dados salvos com sucesso no banco de dados Neon.tech!")

if __name__ == "__main__":
    main()