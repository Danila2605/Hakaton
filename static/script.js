document.getElementById('send-button').addEventListener('click', sendMessage);
document.getElementById('message-input').addEventListener('keypress', function(event) {
    if (event.key === 'Enter') {
        event.preventDefault(); // Отменяет действие по умолчанию
        sendMessage();
    }
});

var messageText = '';

function sendMessage() {
    const input = document.getElementById('message-input');
    messageText = input.value.trim();

    if (messageText) {
        const messagesContainer = document.getElementById('messages');
        const newMessage = document.createElement('div');
        newMessage.className = 'message user';
        newMessage.textContent = messageText;

        const url=`http://localhost:5000/api/data/${messageText}`;
        $.get(url, function (data, status) {
            const responseMessage = document.createElement('div');
            responseMessage.className = 'message'; // Можно добавить класс для стилизации
            responseMessage.innerHTML += 'This problem from \"<b>' + data.predicted_service + "</b>\""
                + "<hr><b>● Solution</b>: " + data.predicted_solution
                + "<br><br><b>● Also you can check this</b>: " + data.predicted_instruction; // Получаем ответ
            messagesContainer.appendChild(responseMessage); // Добавляем ответное сообщение
            messagesContainer.scrollTop = messagesContainer.scrollHeight; // Прокрутка вниз
        });

        messagesContainer.appendChild(newMessage);
        input.value = '';
        messagesContainer.scrollTop = messagesContainer.scrollHeight; // Прокрутка вниз

    }
}

fetch(`http://localhost:5000/api/data/${question}`, {
    cache: "no-cache"
})
    .then(response => response.json())
    .then(data => console.log(data))
    .catch(error => console.error('Error:', error));