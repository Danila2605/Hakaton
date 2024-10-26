import numpy as np

# Функция активации (сигмоид)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Производная сигмоида
def sigmoid_derivative(x):
    return x * (1 - x)

# Загрузка модели для инструкций
loaded_instruction_model = np.load('instruction_model.npz')
weights_input_to_hidden_instruction = loaded_instruction_model['weights_input_to_hidden_instruction']
weights_hidden_to_output_instruction = loaded_instruction_model['weights_hidden_to_output_instruction']
bias_input_to_hidden_instruction = loaded_instruction_model['bias_input_to_hidden_instruction']
bias_hidden_to_output_instruction = loaded_instruction_model['bias_hidden_to_output_instruction']

# Загрузка модели для решений
loaded_solution_model = np.load('solution_model.npz')
weights_input_to_hidden_solution = loaded_solution_model['weights_input_to_hidden_solution']
weights_hidden_to_output_solution = loaded_solution_model['weights_hidden_to_output_solution']
bias_input_to_hidden_solution = loaded_solution_model['bias_input_to_hidden_solution']
bias_hidden_to_output_solution = loaded_solution_model['bias_hidden_to_output_solution']

# Загрузка обученной модели для сервиса
loaded_service_model = np.load('service_model.npz')
weights_input_to_hidden_service = loaded_service_model['weights_input_to_hidden']
weights_hidden_to_output_service = loaded_service_model['weights_hidden_to_output']
bias_input_to_hidden_service = loaded_service_model['bias_input_to_hidden']
bias_hidden_to_output_service = loaded_service_model['bias_hidden_to_output']

# Загрузка данных обучения
loaded_preprocessing = np.load('preprocessing.npz', allow_pickle=True)
vectorizer = loaded_preprocessing['vectorizer'].item()
label_encoder_label = loaded_preprocessing['label_encoder_label'].item()
label_encoder_solution = loaded_preprocessing['label_encoder_solution'].item()
label_encoder_instruction = loaded_preprocessing['label_encoder_instruction'].item()


# Функция для предсказания сервиса
def predict_service(problem):
    # Transform the problem using the same vectorizer
    X_new = vectorizer.transform([problem]).toarray()

    # Ensure the input dimensions match the training data dimensions
    X_new = np.reshape(X_new, (1, -1))  # Reshape to match the input shape expected by the model

    # Прямое распространение для сервиса
    hidden_raw_service = bias_input_to_hidden_service + weights_input_to_hidden_service @ X_new.T
    hidden_service = sigmoid(hidden_raw_service)
    output_raw_service = bias_hidden_to_output_service + weights_hidden_to_output_service @ hidden_service
    output_service = sigmoid(output_raw_service)
    predicted_service = label_encoder_label.inverse_transform([np.argmax(output_service)])[0]

    return predicted_service

# Функция для предсказания решения
def predict_solution(problem):
    # Transform the problem using the same vectorizer
    X_new = vectorizer.transform([problem]).toarray()

    # Ensure the input dimensions match the training data dimensions
    X_new = np.reshape(X_new, (1, -1))  # Reshape to match the input shape expected by the model

    # Прямое распространение для решения
    hidden_raw_solution = bias_input_to_hidden_solution + weights_input_to_hidden_solution @ X_new.T
    hidden_solution = sigmoid(hidden_raw_solution)
    output_raw_solution = bias_hidden_to_output_solution + weights_hidden_to_output_solution @ hidden_solution
    output_solution = sigmoid(output_raw_solution)
    predicted_solution = label_encoder_solution.inverse_transform([np.argmax(output_solution)])[0]

    return predicted_solution

# Функция для предсказания инструкции
def predict_instruction(problem, service, similar_solution):
    # Combine the problem, service, and similar_solution into a single input
    combined_input = f"{problem} {service} {similar_solution}"

    # Transform the combined input using the same vectorizer
    X_new = vectorizer.transform([combined_input]).toarray()

    # Ensure the input dimensions match the training data dimensions
    X_new = np.reshape(X_new, (1, -1))  # Reshape to match the input shape expected by the model

    # Прямое распространение для инструкций
    hidden_raw_instruction = bias_input_to_hidden_instruction + weights_input_to_hidden_instruction @ X_new.T
    hidden_instruction = sigmoid(hidden_raw_instruction)
    output_raw_instruction = bias_hidden_to_output_instruction + weights_hidden_to_output_instruction @ hidden_instruction
    output_instruction = sigmoid(output_raw_instruction)
    predicted_instruction = label_encoder_instruction.inverse_transform([np.argmax(output_instruction)])[0]

    return predicted_instruction

def is_problem(message): 
    problem_keywords = [ 
    "can't", "won't", "haven't", "doesn't", "isn't", "not", "error", "problem", "issue", "crash", "fail", 
    "broken", "malfunction", "bug", "glitch", "fault", "defect", "trouble", "difficulty", 
    "complaint", "concern", "inconvenience", "obstacle", "hurdle", "snag", "hitch", "setback", 
    "challenge", "dilemma", "predicament", "quandary", "mishap", "misfortune", "calamity", 
    "catastrophe", "disaster", "crisis", "emergency", "urgency", "panic", "alarm", "distress", 
    "anxiety", "worry", "fear", "doubt", "uncertainty", "confusion", "perplexity", "bewilderment", 
    "puzzlement", "bafflement", "mystification", "disorientation", "disarray", "chaos", 
    "disorder", "turmoil", "upheaval", "commotion", "agitation", "unrest", "turbulence", 
    "instability", "insecurity", "vulnerability", "fragility", "weakness", "frailty", "delicacy", 
    "sensitivity", "susceptibility", "exposure", "risk", "hazard", "danger", "peril", "threat", 
    "menace", "jeopardy", "endangerment", "imperilment", "precariousness", "insecurity", 
    "vulnerability", "fragility", "weakness", "frailty", "delicacy", "sensitivity", "susceptibility", 
    "exposure", "risk", "hazard", "danger", "peril", "threat", "menace", "jeopardy", "endangerment", 
    "imperilment", "precariousness", "insecurity", "vulnerability", "fragility", "weakness", 
    "frailty", "delicacy", "sensitivity", "susceptibility", "exposure", "risk", "hazard", "danger", 
    "peril", "threat", "menace", "jeopardy", "endangerment", "imperilment", "precariousness" 
] 
    if len(message.split()) == 1 or len(message) <= 8: 
        return False 
    return any(keyword in message.lower() for keyword in problem_keywords)

# Объединяющая функция для предсказания сервиса, решения и инструкции
def predict_all(problem):
    if not is_problem(problem): 
        return "This is not a problem. Please provide a valid problem description.", "Has not been selected yet", "Has not been selected yet"
    
    predicted_service = predict_service(problem)
    predicted_solution = predict_solution(problem)
    predicted_instruction = predict_instruction(problem, predicted_service, predicted_solution)

    return predicted_service, predicted_solution, predicted_instruction


#Пример использования объединяющей функции
# problem = "I can't connect to the internet"
# predicted_service, predicted_solution, predicted_instruction = predict_all(problem)
# print(problem)
# print(f"Service: {predicted_service}")
# print(f"Solution: {predicted_solution}")
# print(f"Instruction: {predicted_instruction}")

# problem = "I haven't received any emails"
# predicted_service, predicted_solution, predicted_instruction = predict_all(problem)
# print(problem)
# print(f"Service: {predicted_service}")
# print(f"Solution: {predicted_solution}")
# print(f"Instruction: {predicted_instruction}")

# problem = "The software keeps crashing"
# predicted_service, predicted_solution, predicted_instruction = predict_all(problem)
# print(problem)
# print(f"Service: {predicted_service}")
# print(f"Solution: {predicted_solution}")
# print(f"Instruction: {predicted_instruction}")

# problem = "How do I use this feature?"
# predicted_service, predicted_solution, predicted_instruction = predict_all(problem)
# print(problem)
# print(f"Service: {predicted_service}")
# print(f"Solution: {predicted_solution}")
# print(f"Instruction: {predicted_instruction}")