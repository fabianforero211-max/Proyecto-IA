import os
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# Carpeta donde están los recortes
CARPETA = r"recortes_16_por_escena_80x80"

class EtiquetadorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🏷️ Etiquetador Rápido (1 a 1) - Barco / No Barco")
        self.root.geometry("480x580")
        self.root.resizable(False, False)

        if not os.path.exists(CARPETA):
            messagebox.showerror("Error", f"No se encontró la carpeta: {CARPETA}")
            self.root.destroy()
            return

        # Cargar archivos que aún no tengan prefijo 1__ o 0__
        todos = sorted([f for f in os.listdir(CARPETA) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.archivos = [f for f in todos if not f.startswith(('1__', '0__'))]

        if len(self.archivos) == 0:
            messagebox.showinfo("Listo", "Todas las imágenes de esta carpeta ya tienen prefijo 1__ o 0__.")
            self.root.destroy()
            return

        self.idx = 0

        # Interfaz Gráfica
        self.lbl_contador = tk.Label(root, text="", font=("Arial", 12, "bold"))
        self.lbl_contador.pack(pady=10)

        self.lbl_archivo = tk.Label(root, text="", font=("Arial", 9), fg="gray")
        self.lbl_archivo.pack()

        # Canvas para mostrar la imagen ampliada (240x240 px para verla con claridad)
        self.lbl_imagen = tk.Label(root)
        self.lbl_imagen.pack(pady=15)

        # Instrucciones
        lbl_instrucciones = tk.Label(
            root, 
            text="Usa el teclado o los botones:\nPresiona [ 1 ] para BARCO | Presiona [ 0 ] para NO BARCO",
            font=("Arial", 10),
            fg="#333333"
        )
        lbl_instrucciones.pack(pady=5)

        # Botones de acción
        frame_btn = tk.Frame(root)
        frame_btn.pack(pady=15)

        self.btn_nobarco = tk.Button(
            frame_btn, text="🔴 NO BARCO (0)", bg="#e74c3c", fg="white", 
            font=("Arial", 12, "bold"), width=15, height=2, command=lambda: self.asignar(0)
        )
        self.btn_nobarco.grid(row=0, column=0, padx=10)

        self.btn_barco = tk.Button(
            frame_btn, text="🟢 BARCO (1)", bg="#27ae60", fg="white", 
            font=("Arial", 12, "bold"), width=15, height=2, command=lambda: self.asignar(1)
        )
        self.btn_barco.grid(row=0, column=1, padx=10)

        # Atajos de teclado (teclado normal y numérico)
        self.root.bind("1", lambda event: self.asignar(1))
        self.root.bind("0", lambda event: self.asignar(0))
        self.root.bind("<KP_1>", lambda event: self.asignar(1))
        self.root.bind("<KP_0>", lambda event: self.asignar(0))

        self.mostrar_imagen()

    def mostrar_imagen(self):
        if self.idx >= len(self.archivos):
            messagebox.showinfo("¡Completado!", "¡Has etiquetado todas las imágenes con éxito!")
            self.root.destroy()
            return

        nombre_actual = self.archivos[self.idx]
        self.lbl_contador.config(text=f"Imagen {self.idx + 1} de {len(self.archivos)}")
        self.lbl_archivo.config(text=nombre_actual)

        ruta_completa = os.path.join(CARPETA, nombre_actual)
        img = Image.open(ruta_completa).convert("RGB")
        # Ampliamos a 240x240 para que sea fácil distinguir el barco
        img_grande = img.resize((240, 240), Image.Resampling.NEAREST)
        
        self.foto_tk = ImageTk.PhotoImage(img_grande)
        self.lbl_imagen.config(image=self.foto_tk)

    def asignar(self, etiqueta):
        nombre_actual = self.archivos[self.idx]
        ruta_vieja = os.path.join(CARPETA, nombre_actual)
        
        nuevo_nombre = f"{etiqueta}__{nombre_actual}"
        ruta_nueva = os.path.join(CARPETA, nuevo_nombre)

        try:
            os.rename(ruta_vieja, ruta_nueva)
        except Exception as e:
            print(f"Error al renombrar: {e}")

        self.idx += 1
        self.mostrar_imagen()

if __name__ == "__main__":
    ventana = tk.Tk()
    app = EtiquetadorApp(ventana)
    ventana.mainloop()