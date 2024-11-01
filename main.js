document.addEventListener("DOMContentLoaded", function() {
    // Search Functionality
    const searchInput = document.querySelector(".search-input");
    const productBoxes = document.querySelectorAll(".shop-sec .box");

    searchInput.addEventListener("input", function() {
        const filter = searchInput.value.toLowerCase();
        productBoxes.forEach(box => {
            const productName = box.querySelector("h2").textContent.toLowerCase();
            if (productName.includes(filter)) {
                box.style.display = ""; // Show the box
            } else {
                box.style.display = "none"; // Hide the box
            }
        });
    });

    // Click Event for "See More" Links
    const seeMoreLinks = document.querySelectorAll(".see-more");
    seeMoreLinks.forEach(link => {
        link.addEventListener("click", function(event) {
            event.preventDefault();
            const productName = link.closest(".box-content").querySelector("h2").textContent;
            alert(`You clicked on 'See More' for: ${productName}`);
        });
    });

    // Language Change Functionality (Demo)
    const languageSelect = document.querySelector(".language");
    languageSelect.addEventListener("change", function() {
        const selectedLanguage = languageSelect.value;
        if (selectedLanguage === "EN") {
            alert("Language changed to English");
            // You can add functionality to change the text on the page here
        } else if (selectedLanguage === "ES") {
            alert("Idioma cambiado a Español");
            // You can add functionality to change the text on the page here
        } else if (selectedLanguage === "HIN") {
            alert("भाषा हिंदी में बदल दी गई");
            // You can add functionality to change the text on the page here
        }
    });
});