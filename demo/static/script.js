$(document).ready(function () {
    $('#qa-form').submit(function (event) {
        event.preventDefault();
        var question = $('#question').val();
        var context = $('#context').val();
        $('#loader').show();
        $.ajax({
            url: '/predict',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                'question': question,
                'context': context
            }),
            success: function (response) {
                $('#loader').hide();
                $('#answer').text(response.answer);
                $('#answer-div').show();
            },
            error: function (xhr, status, error) {
                console.error(xhr.responseText);
                $('#loader').hide();
                alert(
                    'An error occurred while trying to get the answer. Please try again later.');
            }
        });
    });
});