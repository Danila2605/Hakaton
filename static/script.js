document.getElementById('send-button').addEventListener('click', sendMessage);
document.getElementById('message-input').addEventListener('keypress', function(event) {
    if (event.key === 'Enter') {
        event.preventDefault(); // Отменяет действие по умолчанию
        sendMessage();
    }
});

function sendMessage() {
    const input = document.getElementById('message-input');
    const messageText = input.value.trim();

    if (messageText) {
        const messagesContainer = document.getElementById('messages');
        const newMessage = document.createElement('div');
        newMessage.className = 'message user';
        newMessage.textContent = messageText;

        const url='http://localhost:5000/api/data?question = ' + messageText;
        $.getJSON(url, function (data, status) {
            console.log(data.answer[1]);
        });

        messagesContainer.appendChild(newMessage);
        input.value = '';
        messagesContainer.scrollTop = messagesContainer.scrollHeight; // Прокрутка вниз

    }
}

fetch('http://localhost:5000/api/data?question = ""')
    .then(response => response.json())
    .then(data => console.log(data))
    .catch(error => console.error('Error:', error));