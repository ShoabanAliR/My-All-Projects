function predict() {
    document.getElementById("result").innerHTML = "⏳ Predicting...";

    fetch("/predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
            attendance: document.getElementById("attendance").value,
            quiz: document.getElementById("quiz").value,
            assignment: document.getElementById("assignment").value,
            midterm: document.getElementById("midterm").value,
            gpa: document.getElementById("gpa").value
        })
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById("result").innerHTML =
            `🎯 Final Grade: <b>${data.final_grade}</b><br>⚠️ Risk Level: <b>${data.risk}</b>`;
    });
}
