document.addEventListener("DOMContentLoaded", function () {

    // =========================================
    // ELEMENTS
    // =========================================

    const runButton =
        document.getElementById("runButton");

    const kValue =
        document.getElementById("kValue");

    const modelCheckboxes =
        document.querySelectorAll(
            'input[name="model"]'
        );

    const bestModelSection =
        document.getElementById(
            "bestModelSection"
        );

    const bestModel =
        document.getElementById(
            "bestModel"
        );

    const bestAccuracy =
        document.getElementById(
            "bestAccuracy"
        );

    const resultsSection =
        document.getElementById(
            "resultsSection"
        );

    const resultsTable =
        document.getElementById(
            "resultsTable"
        );

    const metricsSection =
        document.getElementById(
            "metricsSection"
        );

    const metricAccuracy =
        document.getElementById(
            "metricAccuracy"
        );

    const metricPrecision =
        document.getElementById(
            "metricPrecision"
        );

    const metricRecall =
        document.getElementById(
            "metricRecall"
        );

    const metricF1 =
        document.getElementById(
            "metricF1"
        );

    const confusionSubtitle =
        document.getElementById(
            "confusionSubtitle"
        );


    // =========================================
    // CSV ELEMENTS
    // =========================================

    const csvFileInput =
        document.getElementById(
            "csvFile"
        );

    const csvInfo =
        document.getElementById(
            "csvInfo"
        );

    const csvFileName =
        document.getElementById(
            "csvFileName"
        );

    const csvSamples =
        document.getElementById(
            "csvSamples"
        );

    const csvFeatures =
        document.getElementById(
            "csvFeatures"
        );

    const targetSection =
        document.getElementById(
            "targetSection"
        );

    const targetColumn =
        document.getElementById(
            "targetColumn"
        );

    const evaluateCsvBtn =
        document.getElementById(
            "evaluateCsvBtn"
        );

    const uploadStatus =
        document.getElementById(
            "uploadStatus"
        );


    // =========================================
    // DATASET PREVIEW ELEMENTS
    // =========================================

    const datasetPreview =
        document.getElementById(
            "datasetPreview"
        );

    const previewTable =
        document.getElementById(
            "previewTable"
        );

    const previewHead =
        document.getElementById(
            "previewHead"
        );

    const previewBody =
        document.getElementById(
            "previewBody"
        );


    // =========================================
    // CHART VARIABLES
    // =========================================

    let accuracyChart = null;
    let foldChart = null;


    // =========================================
    // DISPLAY CONFUSION MATRIX
    // =========================================

    function displayConfusionMatrix(
        matrix,
        modelName
    ) {

        const container =
            document.getElementById(
                "confusionMatrix"
            );

        if (!container) {
            return;
        }

        container.innerHTML = "";

        if (!matrix || matrix.length === 0) {
            return;
        }

        const table =
            document.createElement("table");

        table.className =
            "confusion-table";

        // Header
        const headerRow =
            document.createElement("tr");

        const emptyHeader =
            document.createElement("th");

        emptyHeader.textContent =
            "Actual / Predicted";

        headerRow.appendChild(
            emptyHeader
        );

        for (
            let i = 0;
            i < matrix.length;
            i++
        ) {

            const th =
                document.createElement("th");

            th.textContent =
                "Class " + i;

            headerRow.appendChild(th);
        }

        table.appendChild(
            headerRow
        );


        // Matrix rows
        matrix.forEach(
            function (row, rowIndex) {

                const tr =
                    document.createElement("tr");

                const rowHeader =
                    document.createElement("th");

                rowHeader.textContent =
                    "Class " + rowIndex;

                tr.appendChild(
                    rowHeader
                );


                row.forEach(
                    function (
                        value,
                        colIndex
                    ) {

                        const td =
                            document.createElement("td");

                        td.textContent =
                            value;

                        // Correct prediction
                        if (
                            rowIndex === colIndex
                        ) {

                            td.classList.add(
                                "correct-prediction"
                            );
                        }

                        tr.appendChild(td);
                    }
                );

                table.appendChild(tr);
            }
        );

        container.appendChild(table);
    }


    // =========================================
    // DISPLAY CHARTS
    // =========================================

    function displayCharts(data) {

        const accuracyCanvas =
            document.getElementById(
                "accuracyChart"
            );

        const foldCanvas =
            document.getElementById(
                "foldChart"
            );

        if (
            !accuracyCanvas ||
            !foldCanvas
        ) {
            return;
        }


        // Destroy old charts
        if (accuracyChart) {
            accuracyChart.destroy();
        }

        if (foldChart) {
            foldChart.destroy();
        }


        const modelNames =
            Object.keys(data.results);

        const accuracies =
            modelNames.map(
                function (model) {

                    return (
                        data.results[model]
                            .average_accuracy
                        * 100
                    );
                }
            );


        // =========================================
        // ACCURACY CHART
        // =========================================

        accuracyChart =
            new Chart(
                accuracyCanvas,
                {
                    type: "bar",

                    data: {

                        labels:
                            modelNames,

                        datasets: [

                            {
                                label:
                                    "Average Accuracy (%)",

                                data:
                                    accuracies,

                                borderWidth:
                                    1
                            }

                        ]
                    },

                    options: {

                        responsive: true,

                        maintainAspectRatio:
                            false,

                        scales: {

                            y: {

                                beginAtZero:
                                    true,

                                max:
                                    100
                            }
                        }
                    }
                }
            );


        // =========================================
        // FOLD-WISE LINE CHART
        // =========================================

        const datasets =
            modelNames.map(
                function (model) {

                    return {

                        label:
                            model,

                        data:
                            data.results[model]
                                .fold_scores
                                .map(
                                    function (
                                        score
                                    ) {

                                        return score * 100;
                                    }
                                ),

                        borderWidth:
                            2,

                        tension:
                            0.3
                    };
                }
            );


        const maxFolds =
            Math.max(
                ...modelNames.map(
                    function (model) {

                        return data
                            .results[model]
                            .fold_scores
                            .length;
                    }
                )
            );


        const foldLabels =
            Array.from(
                {
                    length: maxFolds
                },

                function (_, index) {

                    return "Fold " +
                        (index + 1);
                }
            );


        foldChart =
            new Chart(
                foldCanvas,
                {
                    type: "line",

                    data: {

                        labels:
                            foldLabels,

                        datasets:
                            datasets
                    },

                    options: {

                        responsive: true,

                        maintainAspectRatio:
                            false,

                        scales: {

                            y: {

                                beginAtZero:
                                    true,

                                max:
                                    100
                            }
                        }
                    }
                }
            );
    }


    // =========================================
    // DISPLAY RESULTS
    // =========================================

    function displayResults(data) {

        if (!data || !data.results) {
            return;
        }


        // =========================================
        // BEST MODEL
        // =========================================

        if (
            bestModel &&
            bestAccuracy &&
            bestModelSection
        ) {

            bestModel.textContent =
                data.best_model;

            bestAccuracy.textContent =
                (
                    data.best_accuracy * 100
                ).toFixed(2) + "%";

            bestModelSection.classList.remove(
                "hidden"
            );
        }


        // =========================================
        // BEST MODEL METRICS
        // =========================================

        const bestResult =
            data.results[
                data.best_model
            ];


        if (bestResult) {

            if (metricAccuracy) {

                metricAccuracy.textContent =
                    (
                        bestResult
                            .average_accuracy
                        * 100
                    ).toFixed(2) + "%";
            }


            if (metricPrecision) {

                metricPrecision.textContent =
                    (
                        bestResult
                            .precision
                        * 100
                    ).toFixed(2) + "%";
            }


            if (metricRecall) {

                metricRecall.textContent =
                    (
                        bestResult
                            .recall
                        * 100
                    ).toFixed(2) + "%";
            }


            if (metricF1) {

                metricF1.textContent =
                    (
                        bestResult
                            .f1_score
                        * 100
                    ).toFixed(2) + "%";
            }


            if (confusionSubtitle) {

                confusionSubtitle.textContent =
                    data.best_model +
                    " classification results";
            }


            displayConfusionMatrix(
                bestResult.confusion_matrix,
                data.best_model
            );


            if (metricsSection) {

                metricsSection.classList.remove(
                    "hidden"
                );
            }
        }


        // =========================================
        // RESULTS TABLE
        // =========================================

        if (resultsTable) {

            resultsTable.innerHTML = "";

            let index = 1;

            Object.entries(
                data.results
            ).forEach(
                function (
                    [modelName, result]
                ) {

                    const row =
                        document.createElement(
                            "tr"
                        );

                    row.innerHTML = `

                        <td>
                            ${index}
                        </td>

                        <td>
                            ${modelName}
                        </td>

                        <td>
                            ${(
                                result
                                    .average_accuracy
                                * 100
                            ).toFixed(2)}%
                        </td>

                        <td>
                            ${(
                                result
                                    .standard_deviation
                                * 100
                            ).toFixed(2)}%
                        </td>

                        <td>
                            ${
                                modelName ===
                                data.best_model
                                    ? "Best Model"
                                    : "Evaluated"
                            }
                        </td>
                    `;

                    resultsTable.appendChild(
                        row
                    );

                    index++;
                }
            );
        }


        if (resultsSection) {

            resultsSection.classList.remove(
                "hidden"
            );
        }


        // =========================================
        // CHARTS
        // =========================================

        displayCharts(data);
    }


    // =========================================
    // IRIS DATASET EVALUATION
    // =========================================

    if (runButton) {

        runButton.addEventListener(
            "click",
            async function () {

                const selectedModels =
                    Array.from(
                        modelCheckboxes
                    )
                    .filter(
                        function (checkbox) {
                            return checkbox.checked;
                        }
                    )
                    .map(
                        function (checkbox) {
                            return checkbox.value;
                        }
                    );


                if (
                    selectedModels.length === 0
                ) {

                    alert(
                        "Please select at least one model."
                    );

                    return;
                }


                const k =
                    parseInt(
                        kValue.value
                    );


                runButton.disabled =
                    true;

                runButton.textContent =
                    "Evaluating...";


                try {

                    const response =
                        await fetch(
                            "/run",
                            {

                                method:
                                    "POST",

                                headers: {

                                    "Content-Type":
                                        "application/json"
                                },

                                body:
                                    JSON.stringify({

                                        k:
                                            k,

                                        models:
                                            selectedModels
                                    })
                            }
                        );


                    const data =
                        await response.json();


                    if (!response.ok) {

                        throw new Error(
                            data.error ||
                            "Evaluation failed."
                        );
                    }


                    console.log(
                        "Evaluation result:",
                        data
                    );


                    displayResults(
                        data
                    );

                }

                catch (error) {

                    console.error(
                        "Evaluation error:",
                        error
                    );

                    alert(
                        error.message
                    );
                }


                runButton.disabled =
                    false;

                runButton.textContent =
                    "Run Cross-Validation";
            }
        );
    }


    // =========================================
    // CSV FILE UPLOAD
    // =========================================

    if (csvFileInput) {

        csvFileInput.addEventListener(
            "change",
            async function () {

                const file =
                    this.files[0];


                if (!file) {
                    return;
                }


                // Check CSV
                if (
                    !file.name
                        .toLowerCase()
                        .endsWith(".csv")
                ) {

                    uploadStatus.textContent =
                        "Please select a CSV file.";

                    uploadStatus.className =
                        "upload-status error";

                    return;
                }


                uploadStatus.textContent =
                    "Uploading dataset...";

                uploadStatus.className =
                    "upload-status";


                const formData =
                    new FormData();

                formData.append(
                    "file",
                    file
                );


                try {

                    const response =
                        await fetch(
                            "/upload",
                            {

                                method:
                                    "POST",

                                body:
                                    formData
                            }
                        );


                    const data =
                        await response.json();


                    if (
                        !response.ok ||
                        !data.success
                    ) {

                        throw new Error(
                            data.error ||
                            "Upload failed."
                        );
                    }


                    console.log(
                        "Uploaded dataset:",
                        data
                    );


                    // =========================================
                    // DATASET INFORMATION
                    // =========================================

                    if (csvInfo) {
                        csvInfo.classList.remove(
                            "hidden"
                        );
                    }


                    csvFileName.textContent =
                        data.dataset.name;


                    csvSamples.textContent =
                        data.dataset.samples;


                    csvFeatures.textContent =
                        data.dataset.features - 1;


                    // =========================================
                    // TARGET COLUMN OPTIONS
                    // =========================================

                    targetColumn.innerHTML = `

                        <option value="">
                            Select target column
                        </option>

                    `;


                    data.dataset.columns
                        .forEach(
                            function (column) {

                                const option =
                                    document.createElement(
                                        "option"
                                    );

                                option.value =
                                    column;

                                option.textContent =
                                    column;

                                targetColumn
                                    .appendChild(
                                        option
                                    );
                            }
                        );


                    targetSection.classList.remove(
                        "hidden"
                    );


                    evaluateCsvBtn.classList.remove(
                        "hidden"
                    );


                    // =========================================
                    // DATASET PREVIEW
                    // =========================================

                    if (
                        datasetPreview &&
                        previewHead &&
                        previewBody &&
                        data.dataset.preview
                    ) {

                        // Clear old preview
                        previewHead.innerHTML =
                            "";

                        previewBody.innerHTML =
                            "";


                        // =========================================
                        // TABLE HEADER
                        // =========================================

                        const headerRow =
                            document.createElement(
                                "tr"
                            );


                        data.dataset.columns
                            .forEach(
                                function (column) {

                                    const th =
                                        document.createElement(
                                            "th"
                                        );

                                    th.textContent =
                                        column;

                                    headerRow
                                        .appendChild(
                                            th
                                        );
                                }
                            );


                        previewHead.appendChild(
                            headerRow
                        );


                        // =========================================
                        // TABLE ROWS
                        // =========================================

                        data.dataset.preview
                            .forEach(
                                function (row) {

                                    const tr =
                                        document.createElement(
                                            "tr"
                                        );


                                    data.dataset.columns
                                        .forEach(
                                            function (
                                                column
                                            ) {

                                                const td =
                                                    document.createElement(
                                                        "td"
                                                    );

                                                td.textContent =
                                                    row[column] ??
                                                    "";

                                                tr.appendChild(
                                                    td
                                                );
                                            }
                                        );


                                    previewBody
                                        .appendChild(
                                            tr
                                        );
                                }
                            );


                        datasetPreview.classList.remove(
                            "hidden"
                        );
                    }


                    // =========================================
                    // SUCCESS MESSAGE
                    // =========================================

                    uploadStatus.textContent =
                        "Dataset uploaded successfully.";

                    uploadStatus.className =
                        "upload-status success";

                }

                catch (error) {

                    console.error(
                        "CSV upload error:",
                        error
                    );

                    uploadStatus.textContent =
                        error.message;

                    uploadStatus.className =
                        "upload-status error";
                }
            }
        );
    }


    // =========================================
    // CSV DATASET EVALUATION
    // =========================================

    if (evaluateCsvBtn) {

        evaluateCsvBtn.addEventListener(
            "click",
            async function () {

                const selectedTarget =
                    targetColumn.value;


                if (!selectedTarget) {

                    uploadStatus.textContent =
                        "Please select a target column.";

                    uploadStatus.className =
                        "upload-status error";

                    return;
                }


                const file =
                    csvFileInput.files[0];


                if (!file) {

                    uploadStatus.textContent =
                        "Please select a CSV file.";

                    uploadStatus.className =
                        "upload-status error";

                    return;
                }


                const selectedModels =
                    Array.from(
                        document.querySelectorAll(
                            'input[name="model"]:checked'
                        )
                    )
                    .map(
                        function (checkbox) {
                            return checkbox.value;
                        }
                    );


                if (
                    selectedModels.length === 0
                ) {

                    uploadStatus.textContent =
                        "Please select at least one model.";

                    uploadStatus.className =
                        "upload-status error";

                    return;
                }


                const k =
                    parseInt(
                        document.getElementById(
                            "kValue"
                        ).value
                    );


                evaluateCsvBtn.disabled =
                    true;

                evaluateCsvBtn.textContent =
                    "Evaluating...";


                uploadStatus.textContent =
                    "Running K-Fold Cross-Validation...";

                uploadStatus.className =
                    "upload-status";


                try {

                    const response =
                        await fetch(
                            "/run_csv",
                            {

                                method:
                                    "POST",

                                headers: {

                                    "Content-Type":
                                        "application/json"
                                },

                                body:
                                    JSON.stringify({

                                        file:
                                            file.name,

                                        target:
                                            selectedTarget,

                                        k:
                                            k,

                                        models:
                                            selectedModels
                                    })
                            }
                        );


                    const data =
                        await response.json();


                    if (
                        !response.ok ||
                        !data.success
                    ) {

                        throw new Error(
                            data.error ||
                            "CSV evaluation failed."
                        );
                    }


                    console.log(
                        "CSV evaluation result:",
                        data
                    );


                    uploadStatus.textContent =
                        "Evaluation completed successfully.";

                    uploadStatus.className =
                        "upload-status success";


                    displayResults(
                        data
                    );

                }

                catch (error) {

                    console.error(
                        "CSV evaluation error:",
                        error
                    );

                    uploadStatus.textContent =
                        error.message;

                    uploadStatus.className =
                        "upload-status error";
                }


                evaluateCsvBtn.disabled =
                    false;

                evaluateCsvBtn.textContent =
                    "Evaluate Uploaded Dataset";
            }
        );
    }

});
