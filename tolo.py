import cv2

from ultralytics import YOLO





# Carrega o modelo YOLOv8 pré-treinado ('n' significa nano, a versão mais rápida e leve)

model = YOLO('yolov8n.pt')





# Inicia a captura de vídeo da webcam (0 é geralmente a câmera padrão do notebook/PC)

cap = cv2.VideoCapture(0)





if not cap.isOpened():

    print("Erro: Não foi possível acessar a câmera.")

    exit()





print("Pressione a tecla 'q' na janela do vídeo para encerrar o teste.")





while True:

    # Lê um frame (quadro) da câmera

    sucesso, frame = cap.read()





    if sucesso:

        # Roda a inferência do YOLO no frame atual

        # O parâmetro stream=True melhora a performance em vídeos

        resultados = model(frame, stream=True)





        # Como estamos passando um frame por vez, pegamos o primeiro resultado

        for resultado in resultados:

            # O método plot() desenha as bounding boxes e labels diretamente na imagem

            frame_anotado = resultado.plot()





        # Exibe o frame anotado em uma janela

        cv2.imshow("Teste YOLO - Webcam ao Vivo", frame_anotado)





        # Aguarda 1 milissegundo e verifica se a tecla 'q' foi pressionada para sair

        if cv2.waitKey(1) & 0xFF == ord("q"):

            break

    else:

        print("Falha ao capturar a imagem da câmera.")

        break





# Limpeza: libera a câmera e fecha todas as janelas do OpenCV

cap.release()

cv2.destroyAllWindows()