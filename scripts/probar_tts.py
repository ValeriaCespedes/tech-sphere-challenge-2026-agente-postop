from kokoro import KPipeline
import soundfile as sf
import time

LANG_CODE = "e"
VOZ_ESPANOL = "ef_dora"

pipeline = KPipeline(lang_code=LANG_CODE)

texto = "Buenos días, soy su asistente de seguimiento postoperatorio. ¿Cómo se ha sentido desde la cirugía?"

start = time.perf_counter()
generator = pipeline(texto, voice=VOZ_ESPANOL)

for i, (gs, ps, audio) in enumerate(generator):
    sf.write(f"data/prueba_tts_{i}.wav", audio, 24000)
    print(f"Fragmento {i} generado: {gs}")

elapsed = time.perf_counter() - start
print(f"\nLatencia total de síntesis: {elapsed:.2f}s")
print("Archivo guardado en data/prueba_tts_0.wav — ábrelo para escuchar el resultado.")