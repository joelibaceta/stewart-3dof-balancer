# Stewart PPO Agent

Este proyecto implementa un sistema de control basado en Aprendizaje por Refuerzo (RL) para una plataforma de **3 grados de libertad (3DoF)** tipo *Stewart Platform* que debe mantener una bola lo más cerca posible del centro de la superficie.

Está construido sobre PyTorch y Gymnasium, e incluye:
- entorno visual simulado,
- control continuo sobre servos,
- entrenamiento usando **Proximal Policy Optimization (PPO)**.


## Problema a resolver
- **Ambiente**: plataforma robótica tipo Stewart (simulada), 3DOF.
- **Observación**: imagen RGB top-down del entorno.
- **Acciones**: comandos continuos a tres servos.
- **Recompensa**: proporcional a la distancia entre la esfera y el centro de la plataforma.

## Justificación teórica: ¿Por qué usar Actor-Critic?

La elección de un enfoque Actor-Critic se basa en los siguientes criterios, derivados de la teoría de DRL estudiada en clase (ver presentación DRL 5.1):

- Espacio de acción continuo
    - Algoritmos como DQN son inapropiados, pues discretizar el espacio de control para 3 motores sería poco eficiente y escalaría mal.
    - Actor-Critic permite modelar políticas estocásticas sobre acciones continuas.

- Estabilidad y aprendizaje más eficiente
    - Al combinar estimación de valor (Critic) con políticas directas (Actor), se reduce la varianza del gradiente de política.
    - Se evita el alto sesgo de los métodos puramente basados en valor (como TD learning).

- Aprendizaje end-to-end con percepción visual
    - Se utilizan CNNs para procesar directamente las imágenes, lo cual hace viable un aprendizaje desde la percepción hasta el control motor.

## ⚙️ Estructura del proyecto


## 🎯 Objetivo del proyecto

Entrenar un agente RL que, observando imágenes de cámara (visión), aprenda a mover la plataforma para **mantener la bola estable en el centro**.

- Observaciones: stack de 4 imágenes (dim: `12x84x84`)
- Acciones: cambios angulares (`Δθ`) para los 3 servos (`Box(-1, 1)^3` → escalado a radianes)
- Recompensa: función densa basada en:
  - cercanía al centro,
  - penalización por cambios bruscos de acción,
  - estar dentro o fuera del área objetivo.


## 📊 Arquitectura General

### Componentes del Agente

Este agente implementa el algoritmo PPO para entornos con espacios de acción continuos. A continuación, se presenta una tabla que resume sus principales componentes:

| Componente         | Descripción                                                                                       | Ubicación en el código               |
|--------------------|---------------------------------------------------------------------------------------------------|--------------------------------------|
| **ActorCriticCNN** | Red neuronal que integra actor y crítico. El actor predice una distribución gaussiana para muestrear acciones; el crítico estima el valor del estado. | `models/actor_critic_cnn.py`         |
| **PPOAgent**       | Orquesta la interacción con el entorno, recolecta experiencias, administra buffers y coordina el entrenamiento. | `agents/ppo_agent.py`                |
| **PPOTrainer**     | Implementa el algoritmo PPO con Clipped Surrogate Objective y Generalized Advantage Estimation (GAE). | `trainers/ppo_trainer.py`            |
| **Replay Buffer**  | Almacena transiciones `(estado, acción, recompensa, next_state, done)` para entrenamiento por lotes. | Dentro de `ppo_agent.py` (integrado) |
| **Entorno (Gym)**  | Proporciona las observaciones y recompensas en cada paso.   

### CNN Visual

Usamos una arquitectura CNN para extraer características visuales de stacks de frames RGB (input: (12, 84, 84)):

