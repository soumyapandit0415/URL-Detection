.then(response => response.json())
.then(data => {
    console.log("Data received from server:", data); // Add this line
    const resultElement = document.getElementById("result");
    if (resultElement) {
        resultElement.innerText = "Result: " + data.result;
    } else {
        const resultContainer = document.createElement("p");
        resultContainer.className = "result";
        resultContainer.innerText = "Result: " + data.result;
        form.parentNode.insertBefore(resultContainer, form.nextSibling);
    }
})
