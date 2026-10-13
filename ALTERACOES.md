# Alterações no código (12/10/2026)

Este documento lista as alterações feitas no repositório durante a integração e os testes no Gazebo, organizadas por objetivo. As alterações **não foram commitadas**.

> Antes desta sessão o repositório já tinha alterações não commitadas (ex.: `action_adapter_node.py`, `avoidance_logic.py`, `obstacle_localizer_node.py`, `test/`). Este documento cobre apenas o que foi alterado ou criado a partir daí. `package.xml`, `rl_observation_node.py` e `train_fixed_scene.py` aparecem como modificados no `git status`, mas não foram alterados nesta sessão.

## Sumário

| Categoria | Arquivos |
|---|---|
| Correções de integração com o Gazebo | `Dockerfile`, `worlds/obstacle_world.sdf`, `urdf/ur_gz_camera.urdf.xacro`, `action_adapter_node.py`, `rl_policy_node.py` |
| Início e parada de episódio | `action_adapter_node.py`, `launch/vision_bringup.launch.py` |
| Redundância por profundidade | `depth_fallback.py` (novo), `vision_node.py`, `obstacle_localizer_node.py`, `camera_geometry.py` |
| Aprendizado por reforço | `fixed_scene_env.py`, `scene_config.py`, `config/rl_starter.yaml`, `models/ppo_ur5e_scene.zip` (novo) |
| Métricas | `metrics.py`, `metrics_node.py`, `config/scene_ground_truth.yaml` (novos) |
| Dataset e benchmark YOLO | `dataset_labels.py`, `dataset_capture_node.py`, `scripts/benchmark_yolo.py`, `scripts/capture_dataset.sh` (novos), `motion_loop.py` |
| Scripts de experimento | `scripts/run_gazebo_episode.sh`, `scripts/evaluate_kinematic.py`, `scripts/summarize_metrics.py` (novos) |
| Empacotamento e validação | `setup.py`, `validate_project.py` |
| Testes | `test/test_depth_fallback.py`, `test/test_metrics.py`, `test/test_dataset_labels.py` (novos); `test/test_ros_nodes_mocked.py`, `test/test_rl_contract_scene_env.py`, `test/_fake_ros.py` |
| Documentação e resultados | `README.md`, `results/2026-10-12/` (novo) |

---

## 1. Correções de integração com o Gazebo

Problemas que só apareceram ao rodar ROS 2 Jazzy + Gazebo Harmonic de verdade; os testes unitários e os testes com ROS simulado não os detectavam.

### 1.1 `Dockerfile`
- **`ENV PYTHONPATH=/opt/venvs/ros/lib/python3.12/site-packages`** e `ENV YOLO_CONFIG_DIR=/tmp/Ultralytics`.
  - *Problema:* os executáveis gerados pelo colcon usam `#!/usr/bin/python3`, que não enxergava o `ultralytics` instalado só no venv. O `vision_node` morria ao iniciar com `ModuleNotFoundError: No module named 'ultralytics'`.
- **Nova camada com `gymnasium` e `stable-baselines3`**, com `numpy==1.26.4` fixado (exigido pelo `cv_bridge`). Necessária para o `rl_policy_node` rodar no container.
- **Cache da cadeira do Gazebo Fuel no build** (`gz fuel download ... OpenRobotics/models/Chair`).
  - *Problema:* o container é recriado a cada execução e baixava a cadeira toda vez. Uma falha transitória de rede ("Error in REST request") deixava o mundo sem carregar.

### 1.2 `worlds/obstacle_world.sdf`
- **Cadeira movida de `(0.80, 0, 0, yaw 1.57)` para `(0.65, -1.10, 0, yaw 0)`.**
  - *Problema:* a caixa envolvente da cadeira (x ∈ [0,51; 1,09], y ∈ [−0,30; 0,22], z ∈ [0; 0,82]) continha a ponta da ferramenta na pose inicial (0,631; 0,133; 0,374). O braço batia na cadeira ao ir para a pose inicial e o controlador abortava por erro de posição.
- **Nova caixa geométrica `box_obstacle`** (0,10 m de lado, estática e suspensa) em `(0.63, -0.09, 0.37)`, na trajetória da tarefa, na mesma posição do obstáculo da `ur5e_scene`. Não é uma classe COCO: serve para testar a redundância por profundidade.
- **Piso rebaixado** (`<pose>0 0 -0.05 0 0 0</pose>` no `ground_plane`).
  - *Problema:* a caixa do piso, com 0,1 m de espessura e centrada em z = 0, punha a superfície do chão em z = +0,05 m. A cadeira ficava afundada e regiões do chão viravam rastros de obstáculo (erro de localização de até 37 cm).
- Comentário explicando a geometria da cena e a necessidade de manter `config/scene_ground_truth.yaml` sincronizado.

### 1.3 `urdf/ur_gz_camera.urdf.xacro`
- **`camera_mount_joint`: `rpy="0 0 0"` → `rpy="3.14159265359 0 0"`.**
  - *Problema:* no Gazebo a câmera olha ao longo de +X do link e usa +Z como "para cima" da imagem. Com a flange voltada para baixo, a imagem saía de cabeça para baixo e o YOLO não reconhecia a cadeira.
- **`initial_positions` passado para `xacro:ur_ros2_control`**: pose `[0, -1.2, 1.4, -1.7, -1.57, 0]` (primeiro waypoint do `motion_loop`).
  - *Problema:* a pose inicial padrão (braço reto para cima) é singular. O `action_adapter` recusava o episódio (`goal_unreachable`) e, com comandos habilitados, republicava essa pose e impedia mover o braço.

### 1.4 `ur_vision_avoidance/action_adapter_node.py`
- **Removido `point.velocities = qdot` em `build_trajectory`.**
  - *Problema:* o `scaled_joint_trajectory_controller` rejeita trajetórias cujo último ponto tem velocidade diferente de zero ("Velocity of last trajectory point ... is not zero"). Todos os comandos do adaptador eram descartados e o robô nunca se movia.

### 1.5 `ur_vision_avoidance/rl_policy_node.py`
- **Nova função `load_policy()`**, que carrega o PPO com `custom_objects` (espaços de observação e ação, schedules e buffers), sem depender da versão do NumPy.
  - *Problema:* a política foi treinada com NumPy 2.x e o container usa NumPy 1.26. O unpickle falhava (`No module named 'numpy._core.numeric'`) e o nó publicava ação zero, o que na prática virava o modo nominal. Os pesos (torch) são carregados normalmente; conferi que a ação é idêntica nos dois ambientes.

---

## 2. Início e parada explícitos do episódio

### 2.1 `ur_vision_avoidance/action_adapter_node.py`
- Novos parâmetros: `auto_start` (padrão `False`), `start_topic` (`/task/start`) e `stop_topic` (`/task/stop`).
- Novos callbacks `on_start()` (inicia um episódio na pose atual) e `on_stop()` (encerra o episódio).
- Enquanto não houver episódio iniciado, `tick()` publica `/task/state` com `status: "waiting_start"` e **não envia nada ao controlador**, de modo que outras ferramentas podem mover o braço.
- *Problema anterior:* o episódio começava sozinho assim que o launch subia, antes de qualquer verificação ou gravação.

### 2.2 `launch/vision_bringup.launch.py`
- Novos argumentos: `adapter_auto_start`, `enable_metrics`, `metrics_output_dir`, `run_label`, `depth_fallback_enabled` e `depth_fallback_max_m`.
- `rl_model` agora aponta por padrão para a política instalada (`models/ppo_ur5e_scene.zip`).
- Parâmetros novos repassados ao `vision_node` (redundância por profundidade) e ao `action_adapter` (`auto_start`).
- Novo nó `metrics_node`, condicionado a `enable_metrics`.

---

## 3. Redundância por profundidade (detecção sem depender da classe)

*Motivação:* com a cadeira próxima, o YOLO não detectava nada e a câmera terminava embaixo do assento. De perto, o YOLO ainda rotulava a cadeira como "bench", classe que era filtrada.

### 3.1 `ur_vision_avoidance/depth_fallback.py` (novo)
- `find_near_blobs()`: amostra a imagem de profundidade numa grade (`stride`), agrupa células vizinhas mais próximas que `max_range_m` e com diferença de profundidade ≤ `max_depth_step_m` (flood fill 4-conexo), e descarta regiões pequenas (< `min_area_ratio`) ou já cobertas por caixas do YOLO (`exclude_boxes`).
- `DepthBlob`: caixa envolvente, `distance_m` (percentil baixo, conservador, usado na decisão de parada), `area_ratio` e **ponto âncora** (`anchor_uv`, `anchor_depth_m`): centroide da região com a profundidade medida ali, usado na localização 3D.
- Pure NumPy, testável sem ROS; ~17 ms no pior caso.

### 3.2 `ur_vision_avoidance/vision_node.py`
- `"bench"` adicionado às `target_classes` padrão.
- Novos parâmetros: `depth_fallback_enabled` (True), `depth_fallback_max_m` (0,60), `depth_fallback_stride` (8), `depth_fallback_min_area_ratio` (0,02) e `depth_fallback_label` ("unknown").
- A criação de detecções foi refatorada na função interna `add_detection()` (mesma lógica de sobreposição com o caminho e de proximidade para YOLO e profundidade). Cada detecção ganhou o campo `source` (`yolo` ou `depth`); as de profundidade incluem `anchor_uv` e `anchor_depth_m`.
- Regiões de profundidade são desenhadas na imagem anotada (`cv2.rectangle` laranja + rótulo com a distância).
- **Medição de latência** no JSON de `/vision/detections`: `inference_ms` (YOLO), `processing_ms` (callback inteiro) e `age_s` (do carimbo de tempo da imagem até a publicação, em tempo ROS).
- Removida a linha redundante `if not self.require_depth and not math.isfinite(distance_m)`.

### 3.3 `ur_vision_avoidance/obstacle_localizer_node.py`
- Novo parâmetro `min_height_m` (0,03): descarta medições cuja superfície fica abaixo dessa altura no `base_link` (piso). Contador `dropped_below_floor` publicado em `/obstacle/tracks`.
- Quando a detecção tem `anchor_uv`/`anchor_depth_m`, a retroprojeção usa o ponto âncora em vez do centro da caixa com o percentil baixo. Antes, uma faixa de chão ficava ~7 cm acima do piso e escapava do filtro.

### 3.4 `ur_vision_avoidance/camera_geometry.py`
- `detection_to_camera_points()` aceita o novo argumento opcional `anchor_uv`, que substitui o centro da caixa como pixel do ponto de superfície.

---

## 4. Aprendizado por reforço

### 4.1 `ur_vision_avoidance/scene_config.py`
- Novos campos em `SceneConfig`: `reward` (`"legacy"` ou `"shaped"`), `penalty_clearance` (0,16 m), `obstacle_jitter_min` e `obstacle_jitter_max`. Todos são lidos do YAML por `load_scene()`.

### 4.2 `ur_vision_avoidance/fixed_scene_env.py`
- **Randomização da posição do obstáculo** a cada `reset()`, com `self.np_random` (determinística por semente). O `info` do `reset()` passa a incluir `obstacle`.
- **Recompensa extraída para `_reward()`**, com duas variantes:
  - `legacy`: a original, sem mudanças (a `fixed_scene` continua idêntica).
  - `shaped`: `20 × progresso − 0,05` por passo, `−2 × max(0; 0,16 − folga de superfície)`, colisão −30, objetivo +30, tempo esgotado −10.
- *Problema:* com a recompensa original, "parar diante do obstáculo" era um ótimo local (ficar parado custava ~−2 por episódio, tentar passar arriscava −100). A política nunca colidia, mas também nunca chegava.

### 4.3 `config/rl_starter.yaml` (seção `ur5e_scene`)
- `collision_distance_m: 0.12 → 0.18`, alinhado à camada de segurança: raio 0,06 + 0,12; o `action_adapter` bloqueia a 0,10 m da superfície. Antes, a política aprendia rotas que o filtro de segurança proibia no robô.
- `reward: shaped`, `penalty_clearance_m: 0.16`.
- `obstacle_jitter_min_m: [-0.02, -0.06, -0.06]`, `obstacle_jitter_max_m: [0.06, 0.06, 0.06]`.

### 4.4 `models/ppo_ur5e_scene.zip` (novo)
- Política PPO treinada com `train_fixed_scene --scene ur5e_scene --timesteps 1500000 --seed 7`. Com 500k passos ainda não tinha convergido; com 1,5M chega a uma recompensa média de ~35 por episódio.

---

## 5. Métricas por episódio (passo 7 da metodologia)

### 5.1 `config/scene_ground_truth.yaml` (novo)
- Verdade de campo dos obstáculos no `base_link`: caixas alinhadas aos eixos (`center_m`, `size_m`), `label` (classe esperada do detector) e `class_name` (classe no dataset). As dimensões da cadeira vêm da malha do modelo no Fuel.

### 5.2 `ur_vision_avoidance/metrics.py` (novo)
- `box_signed_distance()`: distância com sinal de pontos a uma caixa (negativa dentro).
- `robot_points()`: ponta da ferramenta, pontos dos elos (os mesmos do filtro de segurança) e o corpo da câmera.
- `true_clearance()` e `nearest_box()`.
- `EpisodeRecorder`: acumula um episódio e gera um resumo com:
  - **segurança:** folga real mínima e colisão;
  - **controle:** resultado, tempo até o objetivo, comprimento do trajeto, desvio máximo da linha reta e tempo em cada status;
  - **percepção:** idade das detecções, tempo de inferência, fração de quadros com pedido de parada e detecções por fonte;
  - **localização:** erro da superfície dos rastros em relação à caixa verdadeira mais próxima, total e por rótulo.

### 5.3 `ur_vision_avoidance/metrics_node.py` (novo)
- Assina `/task/state`, `/joint_states`, `/vision/detections` e `/obstacle/tracks`.
- Detecta início e fim de episódio (novo início, `reached`, parada, reinício ou tempo esgotado).
- Publica `/metrics/episode` e grava `episodes.jsonl` e `episodes.csv` em `output_dir`. Campos de chaves variáveis vão como JSON, para o cabeçalho do CSV ser fixo.
- Trata `ExternalShutdownException` no encerramento.

---

## 6. Dataset rotulado automaticamente e benchmark YOLO (Tabela 2)

### 6.1 `ur_vision_avoidance/dataset_labels.py` (novo)
- `project_box()`: projeta uma caixa 3D na imagem (pinhole + TF), recorta nas bordas e calcula `inside_ratio` (truncamento).
- `depth_support()`: fração das amostras de profundidade dentro da caixa 2D que caem na faixa de profundidade do objeto (teste de oclusão).
- `yolo_line()`: linha de rótulo no formato do YOLO.

### 6.2 `ur_vision_avoidance/dataset_capture_node.py` (novo)
- Salva um quadro RGB a cada `period_s`, com rótulos projetados a partir da verdade de campo e da TF no instante da imagem. Descarta objetos ocultos, muito truncados ou pequenos.
- Saída: `images/`, `labels/`, `annotations.jsonl` e `data.yaml` (pronto para ajuste fino com Ultralytics).

### 6.3 `scripts/benchmark_yolo.py` (novo)
- Compara checkpoints nas mesmas imagens: AP50, precisão, revocação e F1 no limiar de operação, revocação independente da classe e latência (média/p95).
- Aceita aliases de classe (`--alias bench=chair`). **Não baixa pesos:** caminhos inexistentes são pulados.

### 6.4 `scripts/capture_dataset.sh` (novo)
- Sobe a simulação, roda o `motion_loop` com waypoints variados (com o tópico de parada remapeado) e o `dataset_capture`. Os waypoints padrão mantêm ≥ 8 cm dos obstáculos verdadeiros, conferido com a cinemática.

### 6.5 `ur_vision_avoidance/motion_loop.py`
- Novo parâmetro opcional `waypoints` (lista achatada de 6×N posições de junta), que substitui os waypoints fixos.

---

## 7. Scripts de experimento

| Script | Função |
|---|---|
| `scripts/run_gazebo_episode.sh` | Roda um episódio sem interface gráfica (`rl`, `potential_field` ou `nominal`). O launch fica num grupo de processos próprio (`setsid`), e o script confere `/joint_states` e as detecções antes de começar, reiniciando a simulação até 3 vezes (a ativação dos controladores às vezes estoura o tempo sob carga). Depois publica `/task/start`, espera a linha de métricas e encerra. Opcional: snapshot da imagem anotada. |
| `scripts/evaluate_kinematic.py` | Compara RL, campo potencial e nominal no laço cinemático do UR5e (60 posições de obstáculo, dentro e fora da faixa de treino), pelo mesmo caminho de dados do ROS. Gera CSV e tabela. |
| `scripts/summarize_metrics.py` | Agrega `episodes.jsonl` numa tabela por modo (média ± desvio). Só usa a biblioteca padrão. |

---

## 8. Empacotamento e validação

### 8.1 `setup.py`
- Instala `config/scene_ground_truth.yaml`, `models/ppo_ur5e_scene.zip` e os scripts em `share/ur_vision_avoidance/scripts`.
- Novos executáveis: `metrics_node` e `dataset_capture`.

### 8.2 `validate_project.py`
- Verifica a existência dos novos arquivos, os executáveis e argumentos de launch novos, e as correções do Gazebo: câmera girada, `initial_positions`, ausência de `point.velocities` e `PYTHONPATH` no Dockerfile.

---

## 9. Testes

Total: **103 testes passando** (eram 79).

| Arquivo | Alteração |
|---|---|
| `test/test_depth_fallback.py` (novo) | Região próxima encontrada com extensão e distância corretas; fundo e profundidade inválida ignorados; ruído pequeno ignorado; objetos em profundidades diferentes não se fundem; exclusão por caixas do YOLO; ponto âncora num piso inclinado; erro de unidade (mm × m). |
| `test/test_metrics.py` (novo) | Distância com sinal; verdade de campo coerente com o mundo; regressão: a pose inicial está livre dos obstáculos e a cadeira antiga a continha; resumo completo de um episódio. |
| `test/test_dataset_labels.py` (novo) | Projeção centrada e tamanho correto; caixa atrás ou fora da imagem descartada; truncamento; oclusão por profundidade; formato do rótulo YOLO. |
| `test/test_ros_nodes_mocked.py` | Ponto de trajetória sem velocidade; sem `auto_start` nada é comandado até `/task/start` (e para com `/task/stop`); filtro de chão; ponto âncora; `metrics_node` gera uma linha por episódio. O helper `make()` usa `auto_start=True` para manter os testes antigos. |
| `test/test_rl_contract_scene_env.py` | `reset(seed=0)` determinístico; randomização limitada, reprodutível por semente e desligada na `fixed_scene`; recompensa `shaped` torna parar pior que avançar. O controlador manual da `ur5e_scene` virou "subir, passar e descer", porque um desvio lateral não cabe na nova distância de colisão. |
| `test/_fake_ros.py` | Adicionado `rclpy.executors.ExternalShutdownException`. |

---

## 10. Documentação e resultados

### 10.1 `README.md`
- Visão geral, tabela de componentes e fluxo de dados atualizados: cena nova, redundância por profundidade, política incluída.
- Configuração do detector corrigida: tolerância de 0,08 s (antes dizia 0,50 s) e percentil 25 da região da caixa (antes dizia "mediana no centro").
- Seção "Run it" atualizada: pose inicial no URDF, `/task/start` e `/task/stop`, pontos sem velocidade.
- Seção de RL: comando de treino da política incluída e o porquê da recompensa nova.
- Nova seção **"Experiments and results"**: tabela dos problemas encontrados no Gazebo, resultados dos episódios no Gazebo, avaliação cinemática e comparação YOLO.
- Lista de arquivos do projeto atualizada.

### 10.2 `results/2026-10-12/` (novo)
- `gazebo_episodes.jsonl` (12 episódios), `kinematic_eval.csv`, `yolo_benchmark.csv`, `annotated_depth_fallback.png`, `dataset_label_check.png` e um `README.md` descrevendo cada arquivo.

---

## Pendências

- **Reconstruir a imagem** (`docker compose build`) para aplicar as mudanças do Dockerfile.
- **Pesos YOLO** (yolo26s, yolo26m, yolo11n, yolov8n): não estão no repositório. Instale-os em `models/` para reproduzir a Tabela 2.
- **Ajuste fino do YOLO para a caixa geométrica:** a captura já gera o dataset no formato certo, mas é preciso mais cenas e pontos de vista.
