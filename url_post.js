const urlForm = document.getElementById("urlForm");
urlForm.addEventListener("submit", function (event) {
    event.preventDefault(); // Prevent form submission
    const url = document.getElementById("urlInput").value;
    // Send URL to backend using AJAX
    $.ajax({
        url: "/classify", // Replace with your backend endpoint
        method: "POST",
        data: { url: url },
        success: function (response) {
            // Handle response from the model
            console.log(response);
            // Display the classification result on the webpage
        },
        error: function (error) {
            console.error(error);
        }
    });
});