import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

# Путь к папке с инструкциями
instruction_folder = r'C:\Python\Dataset\Instructions'

# Чтение файлов инструкций и создание словаря для соответствия проблем и инструкций
instruction_dict = {}
for filename in os.listdir(instruction_folder):
    if filename.endswith('.docx'):
        problem = os.path.splitext(filename)[0]  # Имя файла без расширения
        with open(os.path.join(instruction_folder, filename), 'r', encoding='latin1') as file:
            instruction_dict[problem] = file.read()

# Загрузка датасета из CSV файла с указанием кодировки и пропуском проблемных строк
df = pd.read_csv(r'C:\Python\Dataset\dataset.csv', encoding='latin1', on_bad_lines='skip', sep=';')

# Предобработка данных
vectorizer = TfidfVectorizer()
X_problems = vectorizer.fit_transform(df['Topic']).toarray()
X_instructions = vectorizer.transform(list(instruction_dict.values())).toarray()

# Вычисление косинусного сходства между проблемами и инструкциями
similarity_matrix = cosine_similarity(X_problems, X_instructions)

# Добавление столбца с инструкциями в датасет
df['Instruction'] = [list(instruction_dict.keys())[similarity_matrix[i].argmax()] for i in range(len(df))]

# Проверка загруженных данных
print(df.head())

# Проверка имен столбцов
print(df.columns)

# Использование столбцов 'Topic', 'label', 'Solution' и 'Instruction'
if 'Topic' in df.columns and 'label' in df.columns and 'Solution' in df.columns and 'Instruction' in df.columns:
    X = df['Topic']
    y_label = df['label']
    y_solution = df['Solution']
    y_instruction = df['Instruction']
else:
    raise KeyError("Столбцы 'Topic', 'label', 'Solution' и/или 'Instruction' не найдены в DataFrame")

# Предобработка данных
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X).toarray()

label_encoder_label = LabelEncoder()
y_label = label_encoder_label.fit_transform(y_label)

label_encoder_solution = LabelEncoder()
y_solution = label_encoder_solution.fit_transform(y_solution.dropna())  # Удаляем NaN значения

label_encoder_instruction = LabelEncoder()
y_instruction = label_encoder_instruction.fit_transform(y_instruction)  # Удаляем NaN значения

# Удаление строк с пропущенными значениями в столбце 'Solution' и 'Instruction' для модели решений и инструкций
df_solution = df.dropna(subset=['Solution'])
df_instruction = df.dropna(subset=['Instruction'])

# Проверка, что данные не пусты
if df_solution.empty:
    raise ValueError("Данные для решений пусты после удаления NaN значений")
if df_instruction.empty:
    raise ValueError("Данные для инструкций пусты после удаления NaN значений")

X_solution = df_solution['Topic']
y_solution = df_solution['Solution']

X_instruction = df_instruction['Topic']
y_instruction = df_instruction['Instruction']

# Преобразование данных для модели решений и инструкций
X_solution = vectorizer.transform(X_solution).toarray()
y_solution = label_encoder_solution.transform(y_solution)

X_instruction = vectorizer.transform(X_instruction).toarray()
y_instruction = label_encoder_instruction.transform(y_instruction)

# Разделение данных на обучающую и тестовую выборки для решений и инструкций
X_train_solution, X_test_solution, y_train_solution, y_test_solution = train_test_split(
    X_solution, y_solution, test_size=0.2, random_state=42)

X_train_instruction, X_test_instruction, y_train_instruction, y_test_instruction = train_test_split(
    X_instruction, y_instruction, test_size=0.2, random_state=42)

# Разделение данных на обучающую и тестовую выборки для сервисов
X_train, X_test, y_train, y_test = train_test_split(
    X, y_label, test_size=0.2, random_state=42)

# Функция активации (сигмоид)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Производная сигмоида
def sigmoid_derivative(x):
    return x * (1 - x)

# Параметры нейронной сети для сервисов
input_size = X_train.shape[1]
hidden_size = 50
output_size = len(label_encoder_label.classes_)

weights_input_to_hidden = np.random.uniform(-0.5, 0.5, (hidden_size, input_size))
weights_hidden_to_output = np.random.uniform(-0.5, 0.5, (output_size, hidden_size))
bias_input_to_hidden = np.zeros((hidden_size, 1))
bias_hidden_to_output = np.zeros((output_size, 1))

# Параметры нейронной сети для решений
input_size_solution = X_train_solution.shape[1]
hidden_size_solution = 10
output_size_solution = len(label_encoder_solution.classes_)

weights_input_to_hidden_solution = np.random.uniform(-0.5, 0.5, (hidden_size_solution, input_size_solution))
weights_hidden_to_output_solution = np.random.uniform(-0.5, 0.5, (output_size_solution, hidden_size_solution))
bias_input_to_hidden_solution = np.zeros((hidden_size_solution, 1))
bias_hidden_to_output_solution = np.zeros((output_size_solution, 1))

# Параметры нейронной сети для инструкций
input_size_instruction = X_train_instruction.shape[1]
hidden_size_instruction = 10
output_size_instruction = len(label_encoder_instruction.classes_)

weights_input_to_hidden_instruction = np.random.uniform(-0.5, 0.5, (hidden_size_instruction, input_size_instruction))
weights_hidden_to_output_instruction = np.random.uniform(-0.5, 0.5, (output_size_instruction, hidden_size_instruction))
bias_input_to_hidden_instruction = np.zeros((hidden_size_instruction, 1))
bias_hidden_to_output_instruction = np.zeros((output_size_instruction, 1))

# Изменение гиперпараметров
epochs_labels = 150  # Увеличиваем количество эпох
epochs_solution = 500
epochs_instruction = 500
learning_rate = 0.01  # Уменьшаем скорость обучения

# Обучение определения сервиса у проблемы
# for epoch in range(epochs_labels):
#     e_loss = 0
#     e_correct = 0

#     for i in range(X_train.shape[0]):
#         image = np.reshape(X_train[i], (-1, 1))
#         label = np.zeros((output_size, 1))
#         label[y_train[i]] = 1

#         # Прямое распространение (к скрытому слою)
#         hidden_raw = bias_input_to_hidden + weights_input_to_hidden @ image
#         hidden = sigmoid(hidden_raw)

#         # Прямое распространение (к выходному слою)
#         output_raw = bias_hidden_to_output + weights_hidden_to_output @ hidden
#         output = sigmoid(output_raw)

#         # Вычисление ошибки
#         e_loss += 1 / len(output) * np.sum((output - label) ** 2, axis=0)
#         e_correct += int(np.argmax(output) == np.argmax(label))

#         # Обратное распространение (выходной слой)
#         delta_output = output - label
#         weights_hidden_to_output += -learning_rate * delta_output @ np.transpose(hidden)
#         bias_hidden_to_output += -learning_rate * delta_output

#         # Обратное распространение (скрытый слой)
#         delta_hidden = np.transpose(weights_hidden_to_output) @ delta_output * sigmoid_derivative(hidden)
#         weights_input_to_hidden += -learning_rate * delta_hidden @ np.transpose(image)
#         bias_input_to_hidden += -learning_rate * delta_hidden

#     # Вывод отладочной информации между эпохами
#     print(f"Epoch №{epoch}")
#     print(f"Loss: {round((e_loss[0] / X_train.shape[0]) * 100, 3)}%")
#     print(f"Accuracy: {round((e_correct / X_train.shape[0]) * 100, 3)}%")
#     e_loss = 0
#     e_correct = 0

# # Сохранение модели для сервисов
# np.savez('service_model.npz',
#          weights_input_to_hidden=weights_input_to_hidden,
#          weights_hidden_to_output=weights_hidden_to_output,
#          bias_input_to_hidden=bias_input_to_hidden,
#          bias_hidden_to_output=bias_hidden_to_output)


# # Обучение нейронной сети для решений
# for epoch in range(epochs_solution):
#     e_loss_solution = 0
#     e_correct_solution = 0

#     for i in range(X_train_solution.shape[0]):
#         image = np.reshape(X_train_solution[i], (-1, 1))
#         label_solution = np.zeros((output_size_solution, 1))
#         label_solution[y_train_solution[i]] = 1

#         # Прямое распространение (к скрытому слою)
#         hidden_raw_solution = bias_input_to_hidden_solution + weights_input_to_hidden_solution @ image
#         hidden_solution = sigmoid(hidden_raw_solution)

#         # Прямое распространение (к выходному слою для решения)
#         output_raw_solution = bias_hidden_to_output_solution + weights_hidden_to_output_solution @ hidden_solution
#         output_solution = sigmoid(output_raw_solution)

#         # Вычисление ошибки для решения
#         e_loss_solution += 1 / len(output_solution) * np.sum((output_solution - label_solution) ** 2, axis=0)
#         e_correct_solution += int(np.argmax(output_solution) == np.argmax(label_solution))

#         # Обратное распространение (выходной слой для решения)
#         delta_output_solution = output_solution - label_solution
#         weights_hidden_to_output_solution += -learning_rate * delta_output_solution @ np.transpose(hidden_solution)
#         bias_hidden_to_output_solution += -learning_rate * delta_output_solution

#         # Обратное распространение (скрытый слой для решения)
#         delta_hidden_solution = np.transpose(weights_hidden_to_output_solution) @ delta_output_solution * sigmoid_derivative(hidden_solution)
#         weights_input_to_hidden_solution += -learning_rate * delta_hidden_solution @ np.transpose(image)
#         bias_input_to_hidden_solution += -learning_rate * delta_hidden_solution

#     # Вывод отладочной информации между эпохами
#     print(f"Epoch №{epoch}")
#     print(f"Loss Solution: {round((e_loss_solution[0] / X_train_solution.shape[0]) * 100, 3)}%")
#     print(f"Accuracy Solution: {round((e_correct_solution / X_train_solution.shape[0]) * 100, 3)}%")
#     e_loss_solution = 0
#     e_correct_solution = 0

# # Сохранение модели для решений
# np.savez('solution_model.npz',
#          weights_input_to_hidden_solution=weights_input_to_hidden_solution,
#          weights_hidden_to_output_solution=weights_hidden_to_output_solution,
#          bias_input_to_hidden_solution=bias_input_to_hidden_solution,
#          bias_hidden_to_output_solution=bias_hidden_to_output_solution)


# Обучение нейронной сети для инструкций
# for epoch in range(epochs_instruction):
#     e_loss_instruction = 0
#     e_correct_instruction = 0

#     for i in range(X_train_instruction.shape[0]):
#         image = np.reshape(X_train_instruction[i], (-1, 1))
#         label_instruction = np.zeros((output_size_instruction, 1))
#         label_instruction[y_train_instruction[i]] = 1

#         # Прямое распространение (к скрытому слою)
#         hidden_raw_instruction = bias_input_to_hidden_instruction + weights_input_to_hidden_instruction @ image
#         hidden_instruction = sigmoid(hidden_raw_instruction)

#         # Прямое распространение (к выходному слою для инструкций)
#         output_raw_instruction = bias_hidden_to_output_instruction + weights_hidden_to_output_instruction @ hidden_instruction
#         output_instruction = sigmoid(output_raw_instruction)

#         # Вычисление ошибки для инструкций
#         e_loss_instruction += 1 / len(output_instruction) * np.sum((output_instruction - label_instruction) ** 2, axis=0)
#         e_correct_instruction += int(np.argmax(output_instruction) == np.argmax(label_instruction))

#         # Обратное распространение (выходной слой для инструкций)
#         delta_output_instruction = output_instruction - label_instruction
#         weights_hidden_to_output_instruction += -learning_rate * delta_output_instruction @ np.transpose(hidden_instruction)
#         bias_hidden_to_output_instruction += -learning_rate * delta_output_instruction

#         # Обратное распространение (скрытый слой для инструкций)
#         delta_hidden_instruction = np.transpose(weights_hidden_to_output_instruction) @ delta_output_instruction * sigmoid_derivative(hidden_instruction)
#         weights_input_to_hidden_instruction += -learning_rate * delta_hidden_instruction @ np.transpose(image)
#         bias_input_to_hidden_instruction += -learning_rate * delta_hidden_instruction

#     # Вывод отладочной информации между эпохами
#     print(f"Epoch №{epoch}")
#     print(f"Loss Instruction: {round((e_loss_instruction[0] / X_train_instruction.shape[0]) * 100, 3)}%")
#     print(f"Accuracy Instruction: {round((e_correct_instruction / X_train_instruction.shape[0]) * 100, 3)}%")
#     e_loss_instruction = 0
#     e_correct_instruction = 0

# # Сохранение модели для инструкций
# np.savez('instruction_model.npz',
#          weights_input_to_hidden_instruction=weights_input_to_hidden_instruction,
#          weights_hidden_to_output_instruction=weights_hidden_to_output_instruction,
#          bias_input_to_hidden_instruction=bias_input_to_hidden_instruction,
#          bias_hidden_to_output_instruction=bias_hidden_to_output_instruction)

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


# Проверка на тестовых данных для сервисов
correct_service = 0
for i in range(X_test.shape[0]):
    image = np.reshape(X_test[i], (-1, 1))
    label = np.zeros((output_size, 1))
    label[y_test[i]] = 1

    # Прямое распространение (к скрытому слою для сервиса)
    hidden_raw_service = bias_input_to_hidden_service + weights_input_to_hidden_service @ image
    hidden_service = sigmoid(hidden_raw_service)

    # Прямое распространение (к выходному слою для сервиса)
    output_raw_service = bias_hidden_to_output_service + weights_hidden_to_output_service @ hidden_service
    output_service = sigmoid(output_raw_service)

    if np.argmax(output_service) == np.argmax(label):
        correct_service += 1

print(f"Test Accuracy Service: {round((correct_service / X_test.shape[0]) * 100, 3)}%")

# Проверка на тестовых данных для решений
correct_solution = 0
for i in range(X_test_solution.shape[0]):
    image = np.reshape(X_test_solution[i], (-1, 1))
    label_solution = np.zeros((output_size_solution, 1))
    label_solution[y_test_solution[i]] = 1

    # Прямое распространение (к скрытому слою для решения)
    hidden_raw_solution = bias_input_to_hidden_solution + weights_input_to_hidden_solution @ image
    hidden_solution = sigmoid(hidden_raw_solution)

    # Прямое распространение (к выходному слою для решения)
    output_raw_solution = bias_hidden_to_output_solution + weights_hidden_to_output_solution @ hidden_solution
    output_solution = sigmoid(output_raw_solution)

    if np.argmax(output_solution) == np.argmax(label_solution):
        correct_solution += 1

print(f"Test Accuracy Solution: {round((correct_solution / X_test_solution.shape[0]) * 100, 3)}%")

# Проверка на тестовых данных для инструкций
correct_instruction = 0
for i in range(X_test_instruction.shape[0]):
    image = np.reshape(X_test_instruction[i], (-1, 1))
    label_instruction = np.zeros((output_size_instruction, 1))
    label_instruction[y_test_instruction[i]] = 1

    # Прямое распространение (к скрытому слою для инструкций)
    hidden_raw_instruction = bias_input_to_hidden_instruction + weights_input_to_hidden_instruction @ image
    hidden_instruction = sigmoid(hidden_raw_instruction)

    # Прямое распространение (к выходному слою для инструкций)
    output_raw_instruction = bias_hidden_to_output_instruction + weights_hidden_to_output_instruction @ hidden_instruction
    output_instruction = sigmoid(output_raw_instruction)

    if np.argmax(output_instruction) == np.argmax(label_instruction):
        correct_instruction += 1

print(f"Test Accuracy Instruction: {round((correct_instruction / X_test_instruction.shape[0]) * 100, 3)}%")

# Пример использования модели для предсказания
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

# Пример предсказания
print(predict_instruction("I can't connect to the internet", "Network", "Check your network settings"))
print(predict_instruction("I haven't received any emails", "Email", "Check your email settings"))
print(predict_instruction("The software keeps crashing", "Software", "Reinstall the software"))
print(predict_instruction("How do I use this feature?", "Training", "Read the user manual"))
