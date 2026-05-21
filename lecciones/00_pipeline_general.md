# Pipeline General del Sistema de Benchmarking Cuántico

> **Asignatura**: Trabajo de Fin de Grado — Ingeniería Informática  
> **Tema**: Arquitectura y diseño del sistema de evaluación comparativa de frameworks de simulación cuántica  
> **Nivel**: Grado universitario (cuarto curso)  
> **Prerrequisitos**: fundamentos de computación cuántica (qubits, puertas, medición), Python intermedio, nociones de procesos del sistema operativo

---

## 1. Visión General del Sistema

### 1.1 Qué mide este benchmark

El sistema evalúa ocho frameworks de simulación cuántica —cuatro escritos en Python y cuatro compilados en Rust— bajo condiciones comparables y reproducibles. Las dimensiones de medición son cuatro:

**Tiempo de ejecución.** Se registran los milisegundos de *wall time* que tarda cada framework en ejecutar un circuito desde que se invoca hasta que se obtiene la distribución de resultados. El dato principal que se reporta es la mediana de diez repeticiones independientes, no la media, porque la mediana es robusta frente a *outliers* causados por interrupciones del sistema operativo.

**Memoria residente (RSS).** Se captura el pico de memoria RSS (*Resident Set Size*) del proceso durante cada repetición. Este dato incluye tanto la memoria gestionada por Python como la memoria de extensiones C (simuladores compilados, BLAS, cuBLAS, etc.) que el recolector de basura de Python no controla. Por eso se usa `psutil.Process().memory_info().rss` y no `tracemalloc`, que solo ve asignaciones Python.

**Fidelidad cuántica (JSD).** Una simulación numéricamente incorrecta puede ser rápida sin ser útil. Para detectar errores de implementación y deriva numérica, el sistema compara la distribución de probabilidades obtenida experimentalmente con la distribución teórica ideal mediante la divergencia de Jensen-Shannon. Un JSD cercano a cero indica alta fidelidad; un JSD cercano a uno indica que la simulación ha colapsado a un resultado incorrecto.

**Escalabilidad exponencial.** El número de amplitudes en un simulador de *statevector* crece como 2^n con el número de qubits n. El sistema ajusta una curva `t(n) = α · 2^(β·n)` a los tiempos medidos para distintos valores de n, lo que permite predecir cuándo cada framework se vuelve intractable y comparar la pendiente exponencial entre implementaciones.

### 1.2 Por qué un benchmark cuántico difiere de uno clásico

Los benchmarks de software clásico parten del supuesto de que la misma entrada siempre produce la misma salida en el mismo tiempo. En computación cuántica esto no se cumple por tres razones estructurales:

**Probabilismo inherente.** Una medición cuántica es un proceso estocástico: el mismo circuito ejecutado dos veces puede producir resultados distintos. Por eso el benchmark no mide una única ejecución, sino que realiza `num_shots=1024` disparos del circuito por repetición y construye una distribución empírica de frecuencias. Esta distribución es la que se compara contra el ideal teórico mediante JSD.

**Variabilidad por calentamiento (warm-up).** Los frameworks cuánticos modernos, especialmente los que usan compilación JIT o inicializan grupos de hilos BLAS, exhiben un primer tiempo de ejecución significativamente mayor que los siguientes. Si ese primer tiempo se incluye en las estadísticas, la mediana queda sesgada. El sistema descarta `warmup_runs=1` ejecución antes de comenzar a medir.

**Ruido de planificación del SO.** Los simuladores de circuitos grandes consumen cientos de megabytes de memoria y varios segundos de CPU. Durante ese tiempo, el planificador del sistema operativo puede migrar el proceso entre núcleos, provocar fallos de página o interrumpirlo para atender interrupciones de hardware. Esto introduce *outliers* en las mediciones que la mediana maneja mejor que la media. El coeficiente de variación (CV = σ/μ) indica cuánta dispersión existe: un CV alto (> 0.3) sugiere que las condiciones del sistema durante el benchmark no eran estables.

### 1.3 Las dos capas: Python y Rust

El benchmark cubre dos ecosistemas técnicamente distintos:

**Capa Python** (frameworks Qiskit, Cirq, CUDA-Q, QDisLib): interfaces de alto nivel orientadas a investigación. Los circuitos se definen como objetos Python, se transpilan internamente a representaciones optimizadas, y se ejecutan sobre backends escritos en C, C++ o CUDA. El overhead de Python es real pero pequeño comparado con el coste de simulación para circuitos de más de 7 qubits.

**Capa Rust** (q1tsim, quantr, quantrs2, qcgpu): binarios compilados independientes. Los circuitos se definen en Rust y la simulación corre directamente sin intérprete. Estos binarios se invocan como procesos externos mediante `subprocess.Popen` y devuelven un único objeto JSON por stdout. El tiempo reportado por el propio binario —no el tiempo del proceso Python que lo lanza— es el que se usa para comparar, eliminando así el overhead de arranque del subproceso.

### 1.4 Docker como unidad de despliegue reproducible

La gran amenaza para la reproducibilidad en benchmarking es la diferencia de entorno: versiones de biblioteca, configuración de BLAS, presencia o ausencia de aceleradores, variables de entorno. El sistema usa Docker para fijar todas estas variables. Cada ejecución del benchmark ocurre dentro de un contenedor construido a partir de un `Dockerfile` versionado. La imagen incluye exactamente las versiones de Qiskit, Cirq, CUDA-Q, QDisLib, NumPy, SciPy y Rust especificadas en el archivo de construcción. Ninguna dependencia del sistema anfitrión contamina los resultados, salvo el hardware físico, que es precisamente lo que se quiere medir.

---

## 2. Punto de Entrada: `bench`

El flujo comienza con el script `bench` del sistema anfitrión (no incluido en el repositorio Python, sino en la raíz del proyecto). Su trabajo es preparar el entorno Docker antes de que `run.py` vea un solo qubit.

### 2.1 Detección de GPU, arquitectura y recursos

Antes de construir el comando Docker, `bench` determina:

- **GPU NVIDIA**: invoca `nvidia-smi` para verificar si hay una tarjeta compatible con CUDA. Si la hay, pasa `--gpus all` al comando `docker run`. Si no, elimina ese flag y seleccionará automáticamente plataformas de la familia `*-cpu`.
- **Arquitectura**: `uname -m` distingue `x86_64` de `arm64`/`aarch64`. Esto determina qué imagen Docker se extrae (la imagen `amd64` o la `arm64`) y qué `platform_id` se pasa a `run.py`.
- **RAM disponible**: se calcula una fracción de la RAM total (típicamente el 80%) para pasarla con `--memory` a Docker. Esto evita que el simulador de un circuito de 20 qubits (que puede necesitar 8 GB de statevector) mate el sistema anfitrión.

### 2.2 Lanzamiento del contenedor Docker

El comando resultante tiene esta forma aproximada:

```bash
docker run --rm -it \
  --gpus all \               # solo si GPU detectada
  --memory 12g \             # fracción de RAM del anfitrión
  -e SUPABASE_URL="..." \
  -e SUPABASE_KEY="..." \
  -e BENCH_HOSTNAME="$(hostname)" \
  -v "$(pwd)/results:/app/results" \
  tfg-quantum:latest \
  --platform linux-x86_64-nvidia
```

El flag `BENCH_HOSTNAME` merece atención especial. Dentro del contenedor, `platform.node()` devuelve un identificador generado por Docker (algo como `a3f2c1b0d8e5`) en lugar del hostname real de la máquina. Para que los resultados queden etiquetados con la máquina física que los generó, el script `bench` inyecta el hostname real como variable de entorno, que `hardware.py` lee con prioridad sobre `platform.node()`.

### 2.3 `entrypoint.sh`: selección del `platform_id`

El `entrypoint.sh` del contenedor recibe como argumento el `platform_id` elegido por `bench` y lo reenvía a `run.py`:

```bash
exec uv run python run.py \
  --platform "$PLATFORM_ID" \
  --contributor "$BENCH_CONTRIBUTOR"
```

Los diez `platform_id` posibles están definidos en `run.py` dentro de `PLATFORM_CONFIGS` y cubren todas las combinaciones razonables de sistema operativo, arquitectura y presencia de GPU.

---

## 3. Detección de Hardware (`hardware.py`)

### 3.1 La dataclass `HardwareInfo`

```python
@dataclass
class HardwareInfo:
    hostname: str
    os: str
    os_version: str
    cpu_model: str
    cpu_cores_physical: int
    cpu_cores_logical: int
    cpu_gflops: float
    ram_total_gb: float
    gpu_model: str | None
    gpu_vram_gb: float | None
    python_version: str
```

Todos los campos de `HardwareInfo` se copian en cada resultado del benchmark, de forma que cualquier fila en la base de datos Supabase es completamente autocontenida: no es necesario unirla con ninguna tabla de contexto para saber en qué máquina se generó.

### 3.2 Detección del modelo de CPU

La función `_detect_cpu_model()` implementa una cadena de fallbacks por sistema operativo:

- **Linux x86_64**: lee `/proc/cpuinfo` buscando la línea `model name`.
- **Linux ARM**: `/proc/cpuinfo` no siempre incluye nombre de modelo en ARM; el fallback es `lscpu` buscando la línea `Model name`, y si falla, los campos `Hardware` o `Model` de `/proc/cpuinfo`.
- **macOS**: invoca `sysctl -n machdep.cpu.brand_string`, que devuelve cadenas como `Apple M2 Pro` o `Intel(R) Core(TM) i9-9880H CPU @ 2.30GHz`.
- **Windows**: usa `platform.processor()`.

### 3.3 Por qué se miden GFLOPS: matmul 64×64 como proxy de rendimiento BLAS

El tiempo de simulación de un circuito cuántico depende críticamente del rendimiento de la multiplicación de matrices (BLAS *dgemm*), porque propagar un estado cuántico a través de una puerta de n qubits es equivalente a multiplicar un vector de 2^n amplitudes complejas por una matriz 2^k × 2^k. Por eso la detección de hardware mide el rendimiento numérico real de BLAS con un microbenchmark:

```python
def _measure_cpu_gflops() -> float:
    """64×64 float64 matmul microbenchmark, K=200 iterations."""
    n, K = 64, 200
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    _ = A @ B  # warmup: initialise BLAS thread pool
    t0 = time.perf_counter_ns()
    for _ in range(K):
        _C = A @ B
    elapsed_ns = time.perf_counter_ns() - t0
    if elapsed_ns <= 0:
        return 0.0
    return (2 * n**3 * K) / elapsed_ns * 1000
```

La fórmula `2 * n³ * K / elapsed_ns * 1000` sigue la definición estándar de GFLOPS: una multiplicación de matrices n×n realiza 2n³ operaciones de punto flotante (n³ multiplicaciones + n³ sumas). El factor 1000 ajusta la escala para obtener una cifra legible en el rango de decenas a miles. El primer `A @ B` es el calentamiento del pool de hilos OpenBLAS o MKL; sin ese calentamiento, el primer tiempo sería el doble por la inicialización de los hilos.

Este número no es un GFLOPS de pico del procesador (que requeriría AVX-512 y vectorización perfecta), sino el rendimiento real que el sistema operativo entrega a NumPy en condiciones de benchmark. Es más representativo para predecir el tiempo de simulación.

### 3.4 Detección de GPU

```python
def _detect_gpu() -> tuple[str | None, float | None]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=5,
    )
    if out.returncode != 0 or not out.stdout.strip():
        return None, None
    parts = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]
    name = parts[0]
    vram_mb = float(parts[1])
    return name, vram_mb / 1024.0
```

El sistema solo detecta GPUs NVIDIA porque los únicos frameworks que aprovechan GPU son CUDA-Q (mediante cuStateVec) y qcgpu (mediante OpenCL, que también funciona en NVIDIA). En ausencia de `nvidia-smi`, ambos campos quedan como `None` y se selecciona automáticamente la plataforma `*-cpu`.

### 3.5 El campo `hostname` y la variable `BENCH_HOSTNAME`

La función `detect_hardware` usa un patrón uniforme para todos sus campos: si existe la variable de entorno `BENCH_<CAMPO>` con un valor no vacío, ese valor tiene precedencia sobre la detección automática. Esto permite:

1. Sobreescribir el hostname dentro de Docker (explicado en §2.2).
2. Proporcionar información de hardware que el contenedor no puede detectar automáticamente (por ejemplo, el modelo de CPU en un nodo de clúster donde `/proc/cpuinfo` muestra información genérica del hipervisor).

```python
def _env_or(key: str, default):
    val = os.environ.get(key)
    return val if val and val.strip() else default

def detect_hardware() -> HardwareInfo:
    gpu_model, gpu_vram_gb = _detect_gpu()
    return HardwareInfo(
        hostname=_env_or("BENCH_HOSTNAME", platform.node()),
        os=_normalize_os(_env_or("BENCH_OS", platform.system())),
        ...
    )
```

---

## 4. El Orquestador `run.py`

### 4.1 `PLATFORM_CONFIGS`: las diez plataformas

`run.py` define diez configuraciones de plataforma que mapean cada combinación de OS/arquitectura/GPU a la lista exacta de frameworks que pueden ejecutarse en ella:

```python
PLATFORM_CONFIGS: dict[str, PlatformConfig] = {
    "macos-arm64":           PlatformConfig(frameworks=["qiskit","cirq","cudaq","qdislib",
                                                         "q1tsim","quantr","quantrs2"], ...),
    "macos-x86_64":          PlatformConfig(frameworks=["qiskit","cirq","qdislib",
                                                         "q1tsim","quantr","quantrs2"], ...),
    "linux-x86_64-nvidia":   PlatformConfig(frameworks=["qiskit","cirq","cudaq","qdislib",
                                                         "q1tsim","quantr","quantrs2","qcgpu"],
                                             cudaq_target="nvidia", ...),
    "linux-x86_64-cpu":      PlatformConfig(frameworks=["qiskit","cirq","cudaq","qdislib",
                                                         "q1tsim","quantr","quantrs2"],
                                             cudaq_target="qpp-cpu", ...),
    ...
}
```

Las diferencias más importantes:
- `cudaq_target`: controla si CUDA-Q usa su backend `qpp-cpu` (simulación en CPU mediante la librería C++ Q++) o su backend `nvidia` (GPU con cuStateVec). En plataformas sin GPU NVIDIA, forzar `nvidia` lanzaría un error en tiempo de ejecución.
- `qcgpu` solo aparece en plataformas con `nvidia` porque requiere OpenCL; en macOS está obsoleto por la deprecación de OpenCL desde macOS 10.14.
- `cudaq` se excluye de `macos-x86_64` porque no existen wheels para Intel Mac en PyPI.

### 4.2 El sweep intercalado de Grover y Shor

El bucle principal de `main()` no ejecuta primero todos los tamaños de Grover y luego todos los de Shor, sino que los intercala por tamaño:

```python
for i in range(max_steps):
    n = n_grover_list[i] if i < len(n_grover_list) else None
    N_shor = n_shor_list[i] if i < len(n_shor_list) else None

    if n is not None:
        # Grover con n qubits para todos los frameworks
        ...
    if N_shor is not None:
        # Shor con N = n_shor_list[i] para todos los frameworks
        ...
```

Los valores de n para Grover son `[3, 5, 7, 9, 11]` y los valores de N para Shor son `[15, 21, 35, 77, 143]`. El diseño intercalado tiene dos ventajas:

1. **Checkpoints más frecuentes**: tras cada par (Grover_n, Shor_N) se escribe un archivo JSON parcial. Si el benchmark falla a mitad de ejecución (corte de luz, OOM en el host, timeout), los resultados de los tamaños menores ya están persistidos.

2. **Térmica más realista**: ejecutar primero todos los tamaños grandes de Grover y luego todos los de Shor permitiría que la CPU alcance su temperatura de *throttling* antes de que empiecen las mediciones de Shor. Al intercalar, cada algoritmo experimenta condiciones térmicas similares en cada tamaño.

### 4.3 Lanzamiento de workers Python mediante `subprocess.Popen`

Para cada (framework, algoritmo, n), `run.py` lanza un subproceso Python independiente:

```python
def _run_python_worker(framework, algo, n, config, contributor_name, hw,
                       cudaq_target="qpp-cpu", timeout_s=600.0) -> dict:
    worker_module = f"python.workers.{framework}_worker"
    payload = json.dumps({
        "algo": algo, "n": n, "n_repetitions": config.n_repetitions,
        "num_shots": config.num_shots, "contributor": contributor_name,
        "cudaq_target": cudaq_target,
    })
    proc = subprocess.Popen(
        [sys.executable, "-m", worker_module],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        text=True,
    )
    proc.stdin.write(payload)
    proc.stdin.close()
    ...
```

**Por qué `subprocess` en lugar de `import` directo**: hay tres razones de ingeniería:

1. **Aislamiento de fallos**: si Qiskit lanza una excepción no capturada, genera un *segfault* en su extensión C, o agota la memoria (OOM), ese fallo mata el proceso hijo pero no el orquestador. El orquestador convierte el error en un resultado de tipo `"status": "error"` y continúa con el siguiente framework.

2. **Inicialización limpia**: frameworks como CUDA-Q inicializan el runtime de CUDA al importar el módulo. Si se hiciera `import cudaq` dentro del proceso orquestador y luego se importara Qiskit, los dos runtimes compartirían el mismo proceso con posibles conflictos de memoria GPU. En subprocesos separados, cada framework arranca en memoria virginal.

3. **Medición de startup real**: el tiempo que tarda en importarse un framework y crear su simulador es parte del coste de uso real. Con subprocesos, `startup_time_ms` se mide dentro del worker antes de que el circuito exista.

**Por qué JSON y no pickle o `multiprocessing.Queue`**: JSON es un formato de texto sin estado que puede ser leído por cualquier herramienta, depurado con `cat`, guardado como evidencia y reproducido manualmente. Pickle requiere que sender y receiver tengan exactamente la misma versión de todas las clases implicadas; si el worker corre bajo Python 3.12 y el orquestador bajo 3.11, puede haber incompatibilidades. `multiprocessing.Queue` requiere que ambos procesos sean hijos del mismo proceso Python, lo que no es el caso cuando los binarios Rust son procesos completamente independientes.

### 4.4 Lanzamiento de binarios Rust

Para los frameworks Rust, el orquestador invoca el binario directamente:

```python
def _run_rust_binary(binary, n, target, num_shots, timeout_s=300.0) -> dict:
    proc = subprocess.Popen(
        [str(binary), "--n", str(n), "--target", str(target),
         "--shots", str(num_shots)],
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        text=True,
    )
    ...
    payload = json.loads(lines[-1])
    return payload
```

El binario imprime líneas de progreso en stderr (que hereda el terminal del usuario) y, como última línea en stdout, un único objeto JSON con todos los resultados. El `time_ms` que reporta el binario es el tiempo medido internamente en Rust con su reloj de alta resolución (`std::time::Instant`), no el tiempo que tardó `subprocess.Popen` en ejecutar el proceso. Esta distinción es fundamental: el overhead de Python al lanzar el subproceso (típicamente 50-200 ms) no contamina las mediciones de los frameworks Rust.

### 4.5 El timeout de 600 segundos

```python
try:
    proc.wait(timeout=timeout_s)  # timeout_s = 600.0
except subprocess.TimeoutExpired:
    proc.kill()
    proc.wait()
    return _error_result(framework, algo, n, hw, contributor_name, "timeout")
```

El timeout de 600 segundos (10 minutos) por (framework, algoritmo, n) es deliberadamente generoso porque la simulación de 11 qubits con algunos frameworks puede superar los 5 minutos en hardware modesto. Sin embargo, el timeout evita que un framework bloqueado por un deadlock o un bucle infinito detenga todo el benchmark. La función `_error_result` devuelve un resultado con todos los campos numéricos a cero y `"status": "error"`, que el código de reporting muestra como `ERROR` en la tabla final.

### 4.6 Checkpoints: qué se guarda y cuándo

Tras completar cada tamaño de n, el orquestador escribe dos archivos en `results/`:

- `grover_{timestamp}_n{n}.json`: los resultados de todos los frameworks para ese n concreto.
- `grover_{timestamp}_partial.json`: el documento completo acumulado hasta ese momento.

Cuando el benchmark termina, se escribe el documento final `grover_{timestamp}.json` y se elimina el `_partial.json` (ya obsoleto). Esta estrategia garantiza que incluso un fallo catastrófico al final del benchmark no pierde más de los resultados del último tamaño de n.

En modo Supabase (cuando `SUPABASE_URL` y `SUPABASE_KEY` están definidos), los checkpoints se reemplazan por inserciones directas en la base de datos tras cada tamaño, eliminando la necesidad de archivos locales.

### 4.7 Curva de escalado al final

Al terminar todos los tamaños, el orquestador rellena los campos de escalado de cada resultado:

```python
for result in results:
    fw = result["framework"]
    sd = scaling_by_fw.get(fw, {})            # {n: wall_time_median_ms}
    result["scaling_data"] = {int(k): v for k, v in sd.items()}
    if len(sd) >= 2:
        alpha, beta = fit_scaling_curve(sd)
    else:
        alpha, beta = 0.0, 0.0
    result["scaling_alpha"] = alpha
    result["scaling_beta"] = beta
```

`scaling_by_fw` se construye incrementalmente durante el sweep: cada vez que un framework completa un tamaño con éxito, su tiempo mediano se almacena. Si un framework falló en algunos tamaños, `scaling_by_fw[fw]` tendrá menos de cinco puntos; si tiene al menos dos, el ajuste de curva se intenta igualmente.

---

## 5. El Protocolo de Medición (`benchmark_core.py`)

### 5.1 `BenchmarkConfig`: los parámetros globales

```python
@dataclass
class BenchmarkConfig:
    n_repetitions: int = 30          # Repeticiones para estadísticas
    warmup_runs: int = 1             # Ejecuciones de calentamiento (no se miden)
    n_values: list[int] = field(default_factory=lambda: [3, 5, 7, 9, 11])
    n_values_shor: list[int] = field(default_factory=lambda: [15, 21, 35, 77, 143])
    num_shots: int = 1024            # Shots para distribución empírica
    cpu_sample_interval: float = 0.05  # Intervalo de muestreo de CPU (s)
```

En producción, `run.py` instancia `BenchmarkConfig` con `n_repetitions=10` (no el valor por defecto de 30) para mantener el benchmark en un tiempo razonable. En modo desarrollo (`--dev`), se usa `n_repetitions=1` con un solo valor de n para una prueba rápida de sanidad de menos de dos minutos.

### 5.2 El `_CpuSampler`: hilo daemon de muestreo

```python
class _CpuSampler:
    def __init__(self, interval: float = 0.05):
        self._interval = interval
        self._samples: list[float] = []
        self._stop = threading.Event()
        self._proc = psutil.Process()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._proc.cpu_percent()  # Primer llamado siempre devuelve 0; descartar
        self._thread.start()

    def stop(self) -> float:
        self._stop.set()
        self._thread.join()
        return float(np.mean(self._samples)) if self._samples else 0.0

    def _run(self) -> None:
        while not self._stop.is_set():
            self._samples.append(self._proc.cpu_percent())
            self._stop.wait(self._interval)
```

El hilo es `daemon=True` porque si el proceso principal termina abruptamente (por ejemplo, por un OOM killer del sistema operativo), no queremos que el hilo de muestreo bloquee el cierre del proceso. El intervalo de 50 ms da 20 muestras por segundo: suficiente para capturar variaciones de uso de CPU durante una simulación, sin añadir overhead apreciable al propio proceso bajo medición.

El primer `cpu_percent()` se llama antes de arrancar el hilo y se descarta. Esto es un requisito de `psutil`: la primera llamada a `cpu_percent()` siempre devuelve 0.0 porque establece el punto de referencia; las llamadas posteriores devuelven el porcentaje real desde la última llamada.

### 5.3 `benchmark_run` paso a paso

```python
def benchmark_run(fn, config=None, framework="", algorithm="", n_qubits=0):
    if config is None:
        config = BenchmarkConfig()

    # 1. Warm-up
    for _ in range(config.warmup_runs):
        fn()

    times_ms: list[float] = []
    peak_rss_mb: float = 0.0

    # 2. Loop de repeticiones
    for i in range(config.n_repetitions):
        proc = psutil.Process()
        tracemalloc.start()

        cpu_sampler = _CpuSampler(config.cpu_sample_interval)
        cpu_sampler.start()

        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()

        cpu_mean = cpu_sampler.stop()

        _, peak_traced = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rss_mb = proc.memory_info().rss / 1024 / 1024

        elapsed_ms = (t1 - t0) * 1000.0
        times_ms.append(elapsed_ms)
        peak_rss_mb = max(peak_rss_mb, rss_mb)

    # 3. Estadísticas
    arr = np.array(times_ms)
    median_ms = float(np.median(arr))
    q75, q25 = np.percentile(arr, [75, 25])
    iqr_ms = float(q75 - q25)
    mean_ms = float(np.mean(arr))
    std_ms = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    cv = std_ms / mean_ms if mean_ms > 0 else 0.0
    ...
```

**Paso 1: Warm-up.** Antes de medir, se ejecuta la función una vez sin registrar el tiempo. En Qiskit, esta primera ejecución compila el circuito a instrucciones de bajo nivel y calienta los cachés de memoria. En CUDA-Q, la primera llamada puede inicializar el contexto CUDA. En JVM-based simulators, el JIT compila el bytecode. Sin warm-up, la primera medición sería entre 2× y 100× más lenta que las siguientes, dependiendo del framework.

**Paso 2: Medición con `time.perf_counter`.** `time.perf_counter()` es el reloj de alta resolución de Python, equivalente a `clock_gettime(CLOCK_MONOTONIC)` en Linux y `QueryPerformanceCounter` en Windows. No es afectado por cambios del reloj del sistema (NTP, DST), a diferencia de `time.time()`. La resolución típica es sub-microsegundo en hardware moderno.

**Paso 2: `tracemalloc` y RSS.** Se usan dos instrumentos de memoria simultáneamente porque miden cosas distintas. `tracemalloc` rastrea cada `malloc` y `free` del runtime Python y devuelve el pico de memoria asignado durante el intervalo medido. La RSS de `psutil` incluye todo lo que el proceso tiene mapeado en memoria física: código del ejecutable, bibliotecas compartidas, buffers de GPU (si el driver los mapea en espacio de usuario), y la memoria de extensiones C que no pasa por el allocator Python. Para simuladores cuánticos, la RSS es más representativa porque los statevectors se alojan típicamente en C/C++.

**Paso 3: Estadísticas.** Se calculan mediana, IQR (rango intercuartílico Q75 - Q25), media, desviación estándar con corrección de Bessel (`ddof=1` para estimador insesgado de la varianza poblacional), y coeficiente de variación CV = σ/μ. La razón de reportar CV en lugar de σ sola es que permite comparar la estabilidad de mediciones con escalas de tiempo muy distintas: un framework que tarda 10 ms con σ=1 ms (CV=0.10) es tan estable como uno que tarda 1000 ms con σ=100 ms (CV=0.10).

### 5.4 `measure_build_time`: separar construcción de simulación

```python
def measure_build_time(build_fn: Callable[[], Any], *args: Any) -> float:
    t0 = time.perf_counter()
    build_fn(*args)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0
```

El tiempo de construir el circuito (definir puertas, crear registros, ensamblar el objeto `QuantumCircuit` o equivalente) se mide una sola vez, antes del loop de repeticiones. La razón es que esta operación es determinista y no tiene la variabilidad estadística de la simulación: el resultado de `build_call(n, target)` siempre es el mismo circuito. Medirlo diez veces añadiría ruido sin información.

La separación entre `build_time_ms` y `simulation_time_ms` es crítica para comparar frameworks de forma justa: algunos frameworks (como Qiskit) hacen transpilación y optimización del circuito como parte de la construcción, mientras que otros (como CUDA-Q) hacen la compilación justo antes de ejecutar. Si se midiera únicamente `wall_time_ms` sin desglosar, un framework con compilación costosa pero simulación rápida quedaría penalizado frente a uno que no compila pero simula más despacio.

### 5.5 `compute_jsd`: la divergencia de Jensen-Shannon

```python
def compute_jsd(empirical_dist: dict[str, float],
                theoretical_dist: dict[str, float]) -> float:
    all_states = sorted(set(empirical_dist) | set(theoretical_dist))
    p = np.array([empirical_dist.get(s, 0.0) for s in all_states], dtype=float)
    q = np.array([theoretical_dist.get(s, 0.0) for s in all_states], dtype=float)
    if p.sum() > 0: p /= p.sum()
    if q.sum() > 0: q /= q.sum()
    return float(jensenshannon(p, q))
```

La JSD es una versión simetrizada y acotada de la divergencia de Kullback-Leibler. Para dos distribuciones de probabilidad P y Q con mixtura M = (P+Q)/2:

```
JSD(P‖Q) = ½·KL(P‖M) + ½·KL(Q‖M)
```

donde `KL(P‖M) = Σ P(x) · log(P(x)/M(x))`.

`scipy.spatial.distance.jensenshannon` devuelve la *raíz cuadrada* de la JSD (a veces llamada distancia JS), que toma valores en [0, 1] y satisface la desigualdad triangular, siendo formalmente una métrica.

En el contexto del benchmark, `p` es la distribución empírica obtenida ejecutando el circuito con `num_shots=1024` disparos, y `q` es la distribución teórica. Para el algoritmo de Grover con n qubits y objetivo `target`, la distribución teórica ideal es un estado puro: `q = {format(target, f"0{n}b"): 1.0}`. Un JSD = 0 significa que todos los disparos midieron el estado objetivo, lo que es imposible con shots finitos pero se aproxima cuando el framework implementa correctamente las amplitudes de Grover. Un JSD alto (> 0.3) indica que la simulación está produciendo una distribución casi uniforme —el síntoma típico de una implementación incorrecta del oráculo o de una inversión de fase errónea.

### 5.6 `fit_scaling_curve`: ajuste exponencial

```python
def fit_scaling_curve(scaling_data: dict[int, float]) -> tuple[float, float]:
    ns = np.array(sorted(scaling_data.keys()), dtype=float)
    ts = np.array([scaling_data[int(n)] for n in ns])

    def model(n, alpha, beta):
        return alpha * np.power(2.0, beta * n)

    try:
        popt, _ = curve_fit(model, ns, ts, p0=[1.0, 1.0], maxfev=10000)
        return float(popt[0]), float(popt[1])
    except RuntimeError:
        # Fallback: regresión lineal en log-space
        log_ts = np.log2(ts + 1e-12)
        beta, log_alpha = np.polyfit(ns, log_ts, 1)
        return float(2.0**log_alpha), float(beta)
```

El modelo `t(n) = α · 2^(β·n)` es la forma natural del escalado de un simulador de statevector: almacenar el estado de n qubits requiere 2^n amplitudes complejas, y propagar el estado a través de una puerta de k qubits requiere una multiplicación de matrices 2^k × 2^k. Para un simulador de statevector puro sin optimizaciones de circuito, β debería ser exactamente 1.

Un β > 1 indica peor que exponencial (por ejemplo, cuando el simulador reconstruye el statevector completo por cada puerta en lugar de actualizar solo los elementos afectados). Un β < 1 indica mejor que exponencial (por ejemplo, cuando el simulador explota esparsidad o comprensión de estado), lo que es muy raro en simuladores de propósito general.

`scipy.optimize.curve_fit` usa el algoritmo de Levenberg-Marquardt para el ajuste no lineal de mínimos cuadrados. Si no converge en 10000 evaluaciones, el código cae al fallback: una regresión lineal en espacio logarítmico (`log2(t) = log2(α) + β·n`), que es equivalente al ajuste exponencial cuando los errores de medición son pequeños.

---

## 6. El Patrón Worker (`workers/_base.py`)

### 6.1 La interfaz stdin/stdout JSON

Cada worker es un módulo Python ejecutable como `python -m python.workers.qiskit_worker`. Su protocolo de comunicación con el orquestador es intencionalmente simple:

```python
def read_config() -> dict:
    return json.loads(sys.stdin.read())

def write_result(result: dict) -> None:
    print(json.dumps(result), flush=True)

def write_error(message: str) -> None:
    print(json.dumps({"status": "error", "error": message}), flush=True)
    sys.exit(1)
```

La robustez de este diseño frente a fallos es máxima porque:

- Si el worker muere por un `SegmentationFault` en una extensión C, el proceso simplemente termina sin stdout. El orquestador detecta el código de salida no-cero y construye un `_error_result`.
- Si el worker agota la memoria, el OOM killer del kernel lo termina enviando SIGKILL. El orquestador detecta el código de salida negativo (en Linux, `-9` para SIGKILL) e informa el error.
- Si el worker produce JSON malformado, `json.loads(lines[-1])` lanzará `JSONDecodeError`, que el orquestador captura y convierte en error.

El `flush=True` en `write_result` es necesario porque stdout en modo texto está típicamente con buffer de línea. Sin flush, el JSON podría quedarse en el buffer del proceso hijo y no llegar al orquestador antes de que el proceso termine.

### 6.2 `run_grover_worker` y `run_shor_worker`

Las dos funciones de `_base.py` son el punto de entrada que cada worker de framework concreto (p.ej. `qiskit_worker.py`) llama después de inicializar su framework específico:

```python
def run_grover_worker(framework, n, config, hw, contributor,
                      startup_ms, search_call, build_call) -> dict:
    target = n
    build_ms = measure_build_time(lambda: build_call(n, target))
    result = benchmark_run(
        lambda: search_call(n, target, config.num_shots),
        config, framework=framework, algorithm="grover", n_qubits=n,
    )
    ...
    result.simulation_time_ms = max(0.0, result.wall_time_median_ms - build_ms)
    ...
```

Los argumentos `search_call` y `build_call` son callables que encapsulan la lógica específica de cada framework. Cada `*_worker.py` implementa estas dos funciones según la API de su framework y las pasa a `run_grover_worker`. Esto permite que `_base.py` sea completamente agnóstico del framework: no importa ninguna biblioteca cuántica.

### 6.3 La jerarquía de tiempos: cuatro dimensiones temporales

La decisión de diseño más importante del sistema es la separación del tiempo total en cuatro componentes ortogonales:

**`startup_time_ms`**: tiempo de inicializar el framework una sola vez al arrancar el worker. Incluye los imports de Python (`import qiskit`, `import cirq`), la inicialización del contexto CUDA (si aplica), la creación del simulador o backend (`AerSimulator()`, `cirq.Simulator()`), y cualquier carga de tablas de puertas o compilación JIT que ocurra en ese momento. Se mide una sola vez con `measure_startup_time()` antes del loop de repeticiones.

**`build_time_ms`**: tiempo de construir el objeto circuito para un n dado. Incluye la creación de registros de qubits, la adición de puertas y la transpilación si el framework la hace en este paso. Se mide una sola vez antes del loop con `measure_build_time()`.

**`simulation_time_ms`**: tiempo de ejecutar la simulación del estado cuántico y tomar shots. Se estima como `wall_time_median_ms - build_time_ms`. Para los binarios Rust, es el tiempo reportado directamente por el binario (medido con `std::time::Instant` desde Rust).

**`wall_time_ms`** (por repetición): tiempo total de `build + simulation` medido en el loop de `benchmark_run`. Es lo que mide `time.perf_counter`. La suma de `build_time_ms + simulation_time_ms` debería aproximarse a este valor; la diferencia representa overhead de Python (llamadas de función, GIL, etc.).

Esta jerarquía permite que comparaciones justas entre frameworks sean posibles. Por ejemplo: si Qiskit tarda 800 ms en construir un circuito de 11 qubits (transpilación pesada) pero solo 50 ms en simularlo, y un framework más simple tarda 5 ms en construirlo pero 200 ms en simularlo, la comparación de `wall_time_ms` sería engañosa. La comparación de `simulation_time_ms` revela cuál simulador es más eficiente en el núcleo de la computación.

---

## 7. Formato de Resultados

### 7.1 Estructura del JSON de checkpoint

Cada checkpoint (y el documento final) tiene la siguiente estructura de alto nivel:

```json
{
  "schema_version": "1.0",
  "generated_at": "2024-11-15T10:23:45.123456+00:00",
  "python_version": "3.12.0 (main, ...)",
  "platform": "Linux-6.1.0-amd64-x86_64-with-glibc2.35",
  "platform_id": "linux-x86_64-nvidia",
  "gpu_enabled": true,
  "emulated": false,
  "no_gpu": false,
  "benchmark_image": "tfg-quantum:v1.2.0",
  "contributor_name": "Ana García",
  "hardware": {
    "hostname": "servidor-lab-03",
    "os": "linux",
    "os_version": "...",
    "cpu_model": "Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz",
    "cpu_cores_physical": 16,
    "cpu_cores_logical": 32,
    "cpu_gflops": 342.7,
    "ram_total_gb": 64.0,
    "gpu_model": "NVIDIA A100-SXM4-40GB",
    "gpu_vram_gb": 40.0,
    "python_version": "3.12.0"
  },
  "config": {
    "n_repetitions": 10,
    "n_values": [3, 5, 7, 9, 11],
    "n_values_shor": [15, 21, 35, 77, 143],
    "num_shots": 1024
  },
  "results": [...]
}
```

### 7.2 `scaling_alpha`, `scaling_beta` y `scaling_data`

Cada resultado individual incluye tres campos relacionados con la escalabilidad:

- `scaling_data`: diccionario `{n: wall_time_median_ms}` con los tiempos medianos para cada valor de n donde el framework completó con éxito.
- `scaling_alpha`: el coeficiente α del modelo `t(n) = α · 2^(β·n)`. Su valor absoluto (en ms) depende de la velocidad absoluta del sistema y no es comparable entre máquinas. Dentro de una misma máquina, un α menor indica un framework más rápido en el punto de referencia n=0 (extrapolado).
- `scaling_beta`: el exponente β. Este es el campo más importante para comparación inter-plataforma porque está adimensionalizado: un β cercano a 1 indica que el framework escala como la teoría predice para simulación de statevector; un β significativamente mayor que 1 indica overhead superexponencial que sugiere ineficiencias algorítmicas.

### 7.3 JSD como indicador de correctitud

El JSD tiene dos usos en el sistema:

1. **Control de calidad**: si un framework reporta JSD > 0.5, sus resultados de rendimiento son sospechosos porque la simulación no converge al estado correcto. Esto puede indicar un bug en la implementación del algoritmo para ese framework, un problema de precisión numérica (acumulación de errores de punto flotante), o una incompatibilidad de fase entre la convención del sistema y la del benchmark.

2. **Comparación de fidelidad**: entre frameworks correctos (JSD bajo), la diferencia de JSD indica qué framework reproduce la distribución ideal con más precisión. Esto es relevante cuando se comparan backends que usan diferentes técnicas numéricas (float32 vs float64, álgebra real vs compleja, etc.).

### 7.4 Integración con Supabase

En modo producción (con `SUPABASE_URL` y `SUPABASE_KEY` definidos), los resultados se envían directamente a Supabase en lugar de escribir archivos JSON locales:

```python
def _supabase_insert(rows: list[dict], url: str, key: str) -> bool:
    endpoint = f"{url}/rest/v1/benchmark_runs"
    payload = json.dumps(rows, default=str).encode()
    req = urllib.request.Request(
        endpoint, data=payload,
        headers={"Content-Type": "application/json",
                 "apikey": key, "Authorization": f"Bearer {key}",
                 "Prefer": "return=minimal"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        print(f"  → Supabase: {len(rows)} filas insertadas ({resp.status})")
```

La función `_expand_result_to_rows` expande cada resultado en una fila por repetición: si un framework completó `n_repetitions=10`, se insertan 10 filas con `repetition_index` de 0 a 9 y `wall_time_ms` con el tiempo individual de cada repetición. Esto permite análisis estadísticos completos en el backend (distribuciones, tests de hipótesis, detección de outliers) sin que el cliente tenga que reenviar los datos raw.

---

## 8. Docker y Reproducibilidad

### 8.1 Por qué Docker y no virtualenv

Un virtualenv fija las versiones de los paquetes Python, pero no fija:

- La versión de la biblioteca C (glibc)
- La versión de CUDA y sus drivers
- El compilador GCC/Clang usado para compilar extensiones C
- Las flags de optimización del compilador (que afectan directamente al rendimiento)
- Las variables de entorno que configuran OpenBLAS (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`)
- La configuración del kernel (tamaño de página, política del planificador)

Docker no puede controlar el kernel ni los drivers del hardware, pero sí fija todo lo demás. La imagen Docker es una snapshot del filesystem completo del entorno de ejecución, construida de forma reproducible a partir de un `Dockerfile` versionado.

### 8.2 El Dockerfile multi-stage

El `Dockerfile` del proyecto usa un build multi-stage para minimizar el tamaño de la imagen final y separar las fases de construcción de la fase de ejecución:

**Stage `qcgpu-amd64`**: cross-compilación del crate `qcgpu` para amd64 usando QEMU. Este stage existe porque `qcgpu` depende de una versión antigua de `ocl-core` con código que el compilador Rust moderno rechaza; se usa un fork vendorizado en `rust/qcgpu/vendor/ocl-core` para poder compilar.

**Stage `rust-builder`**: compila los cuatro crates Rust (`q1tsim-grover`, `quantr-grover`, `quantrs2-grover`, `qcgpu-grover` y sus variantes Shor) con `cargo build --release`. La compilación en release activa todas las optimizaciones de LLVM (`-O3`, vectorización, inlining agresivo), lo que puede dar 5-10× mejor rendimiento que debug.

**Base amd64** (`nvidia/cuda:12.x-base`): para plataformas Linux x86_64 con GPU, la imagen base es la imagen oficial CUDA de NVIDIA. Esto garantiza la presencia de `libcuda.so`, `libcudart.so`, y las herramientas como `nvidia-smi` que el código de detección de GPU necesita. Sin esta imagen base, CUDA-Q no puede cargar cuStateVec.

**Base arm64** (`python:3.12-slim`): para plataformas sin GPU NVIDIA (macOS, Linux ARM sin GPU, Windows), la imagen base es la imagen oficial de Python que no incluye CUDA. Esto reduce el tamaño de la imagen de ~8 GB a ~2 GB.

**Stage `python-deps`**: instala todas las dependencias Python usando `uv` en lugar de `pip`. `uv` es entre 10× y 100× más rápido que `pip` para resolver e instalar dependencias porque está escrito en Rust y usa una estrategia de resolución de dependencias diferente. En un pipeline de CI donde la imagen se construye frecuentemente, esta diferencia es significativa.

**Stage de runtime final**: copia los binarios Rust del stage `rust-builder` y las dependencias Python del stage `python-deps` en una imagen mínima. El resultado es una imagen que contiene exactamente lo necesario para ejecutar el benchmark, sin los SDKs de compilación que son necesarios para construir pero no para ejecutar.

### 8.3 Dependencias opcionales en plataformas soportadas

CUDA-Q y QDisLib no están disponibles en PyPI para todas las plataformas. CUDA-Q solo tiene wheels para Linux x86_64 y Linux aarch64; en macOS Intel no existen wheels y el intento de instalación falla con un error de plataforma no soportada. QDisLib requiere MPI, que en Docker necesita configuración especial del entorno de red.

Por esta razón, el `Dockerfile` instala CUDA-Q y QDisLib condicionalmente según el `--build-arg PLATFORM` pasado durante `docker build`. Si la plataforma no los soporta, la imagen no los incluye, y `PLATFORM_CONFIGS` en `run.py` tampoco los lista para esa plataforma.

---

## 9. Diagrama del Flujo Completo

```
bench (script anfitrión)
    │
    ├─ Detectar GPU (nvidia-smi)
    ├─ Detectar arquitectura (uname -m)
    ├─ Calcular RAM disponible
    └─ docker run tfg-quantum --platform <id>
         │
         └─ entrypoint.sh → uv run python run.py --platform <id>
              │
              ├─ detect_hardware()          [hardware.py]
              │    ├─ _detect_cpu_model()
              │    ├─ _measure_cpu_gflops()
              │    └─ _detect_gpu()
              │
              ├─ PLATFORM_CONFIGS[platform]  [run.py]
              │
              └─ bucle intercalado Grover/Shor
                   │
                   ├─ para cada (n_grover, N_shor):
                   │    │
                   │    ├─ para cada framework Python:
                   │    │    └─ subprocess.Popen python.workers.<fw>_worker
                   │    │         │  stdin: JSON config
                   │    │         │  stderr: progress → terminal
                   │    │         └─ stdout: JSON result
                   │    │              │
                   │    │              ├─ read_config()        [_base.py]
                   │    │              ├─ startup_time_ms
                   │    │              ├─ run_grover_worker()
                   │    │              │    ├─ measure_build_time()  [benchmark_core]
                   │    │              │    ├─ benchmark_run()       [benchmark_core]
                   │    │              │    │    ├─ warmup
                   │    │              │    │    ├─ loop × n_rep:
                   │    │              │    │    │    ├─ perf_counter
                   │    │              │    │    │    ├─ tracemalloc
                   │    │              │    │    │    └─ _CpuSampler
                   │    │              │    │    └─ estadísticas
                   │    │              │    └─ compute_jsd()         [benchmark_core]
                   │    │              └─ write_result(JSON)
                   │    │
                   │    └─ para cada framework Rust:
                   │         └─ subprocess.Popen <binary> --n --target --shots
                   │              └─ stdout: JSON result (time_ms interno)
                   │
                   └─ checkpoint JSON / Supabase insert
                        │
                        └─ tras todos los n:
                             ├─ fit_scaling_curve()  [benchmark_core]
                             └─ JSON final / Supabase patch scaling
```

---

## 10. Consideraciones Avanzadas y Limitaciones

### 10.1 Variabilidad entre ejecuciones

Incluso con 10 repeticiones, los tiempos de simulación cuántica presentan variabilidad inter-ejecución (entre ejecuciones separadas del benchmark completo, en lugar de entre repeticiones dentro de una misma ejecución). Las causas principales son:

- **Temperatura de la CPU**: la frecuencia Turbo Boost depende de la temperatura. Una CPU que ejecuta el benchmark por segunda vez puede estar 10°C más caliente y funcionar a una frecuencia 5-10% menor.
- **NUMA y afinidad de caché**: en sistemas multi-socket, el planificador del kernel puede asignar el proceso a un socket diferente entre ejecuciones, cambiando la latencia de acceso a memoria.
- **Estado del driver CUDA**: algunos estados de inicialización CUDA no se reinician entre ejecuciones del mismo binario.

El coeficiente de variación CV captura la variabilidad intra-ejecución pero no la inter-ejecución. Para comparaciones rigurosas, se recomienda ejecutar el benchmark en condiciones de carga mínima del sistema y comparar medianas de múltiples ejecuciones completas.

### 10.2 El problema de q1tsim y qcgpu

`q1tsim` es un crate abandonado desde 2019 que depende de versiones antiguas de `ndarray` y `rand`. Para poder compilarlo con la toolchain Rust moderna, el proyecto lo vendoriza en `rust/q1tsim/vendor/` con el tipo de crate cambiado de `dylib` a `rlib`. Esto elimina la dependencia en `libq1tsim.so` que de otro modo sería necesaria en el contenedor de runtime.

`qcgpu` requiere OpenCL. En macOS, OpenCL está marcado como obsoleto desde macOS 10.14 y Apple recomienda Metal. Los resultados de `qcgpu` en macOS pueden tener mayor variabilidad que en Linux con GPU NVIDIA, donde el driver OpenCL de NVIDIA es estable.

### 10.3 Modo emulación QEMU

Cuando se ejecuta una imagen Docker amd64 en un sistema anfitrión arm64 (por ejemplo, una MacBook M-series), Docker usa QEMU para traducir las instrucciones x86_64 a ARM. CUDA-Q compila con instrucciones AVX/SSE que QEMU no emula, produciendo una señal SIGILL (instrucción ilegal). El flag `--emulated` del orquestador excluye automáticamente CUDA-Q en este escenario:

```python
if args.emulated:
    _qemu_broken = {"cudaq"}
    skipped_e = [fw for fw in all_enabled if fw in _qemu_broken]
    all_enabled = [fw for fw in all_enabled if fw not in _qemu_broken]
```

---

## 11. Referencias

1. **Nielsen, M. A. & Chuang, I. L.** (2010). *Quantum Computation and Quantum Information* (10th anniversary ed.). Cambridge University Press. — Fundamento teórico de los algoritmos de Grover y Shor implementados en el benchmark.

2. **Georgiou, K., Niknami, A., & Luk, W.** (2023). Benchmarking quantum simulation. *IEEE Transactions on Quantum Engineering*, 4, 1–14. https://doi.org/10.1109/TQE.2023.3243701 — Marco metodológico para evaluación comparativa de simuladores cuánticos.

3. **Fingerhuth, M., Babej, T., & Wittek, P.** (2018). Open source software in quantum computing. *PLOS ONE*, 13(12), e0208561. https://doi.org/10.1371/journal.pone.0208561 — Revisión del ecosistema de software cuántico de código abierto que motiva la selección de frameworks.

4. **Aleksandrowicz, G., et al.** (2019). Qiskit: An open-source framework for quantum computing. *Zenodo*. https://doi.org/10.5281/zenodo.2562110 — Paper de referencia de Qiskit.

5. **Cirq Developers** (2024). Cirq: A Python library for writing, manipulating, and optimizing quantum circuits. https://github.com/quantumlib/Cirq — Repositorio y documentación principal de Cirq.

6. **Baydin, A. G., et al.** (2018). Automatic differentiation in machine learning: a survey. *Journal of Machine Learning Research*, 18(153), 1–43. — Contexto sobre JIT y compilación en sistemas de computación de alto rendimiento relevante para la sección de warm-up.

7. **Fleming, P. J. & Wallace, J. J.** (1986). How not to lie with statistics: the correct way to summarize benchmark results. *Communications of the ACM*, 29(3), 218–221. https://doi.org/10.1145/5666.5673 — Justificación clásica del uso de mediana e IQR frente a media en benchmarking.

8. **Lin, J.** (1991). Divergence measures based on the Shannon entropy. *IEEE Transactions on Information Theory*, 37(1), 145–151. https://doi.org/10.1109/18.61115 — Paper original que define la divergencia Jensen-Shannon usada como métrica de fidelidad.

9. **Guerreschi, G. G., et al.** (2020). Intel Quantum Simulator: A cloud-ready high-performance simulator of quantum circuits. *Quantum Science and Technology*, 5(3), 034007. https://doi.org/10.1088/2058-9565/ab8505 — Describe técnicas de optimización de simuladores de statevector relevantes para interpretar los valores de α y β del ajuste de escalado.

10. **Smelyanskiy, M., Sawaya, N. P. D., & Aspuru-Guzik, A.** (2016). qHiPSTER: The quantum high performance software testing environment. *arXiv:1601.07195*. — Análisis del escalado exponencial de simuladores de statevector en HPC; fundamento de la curva `t(n) = α · 2^(β·n)`.

11. **Merkel, D.** (2014). Docker: Lightweight Linux containers for consistent development and deployment. *Linux Journal*, 2014(239), 2. — Artículo que popularizó Docker como herramienta de reproducibilidad en software engineering, justificando su uso en benchmarking científico.

12. **Boixo, S., et al.** (2018). Characterizing quantum supremacy in near-term devices. *Nature Physics*, 14(6), 595–600. https://doi.org/10.1038/s41567-018-0124-x — Define metodología de benchmarking de circuitos aleatorios en dispositivos cuánticos; el diseño del sistema de este TFG adapta sus principios al dominio de simulación clásica.

---

*Documento generado el 22 de mayo de 2026. Basado en el código fuente de `run.py`, `python/benchmark_core.py`, `python/hardware.py` y `python/workers/_base.py` del repositorio TFG.*
