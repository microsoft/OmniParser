window.HELP_IMPROVE_VIDEOJS = false;

$(document).ready(function() {
    // Check for click events on the navbar burger icon

    var options = {
        slidesToScroll: 1,
        slidesToShow: 1,
        loop: true,
        infinite: true,
        autoplay: true,
        autoplaySpeed: 5000,
    }

    // Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    bulmaSlider.attach();

})


$(document).ready(function() {
    var trs = $('#tabResults').children("[class!=th]")
    for (let item of trs) {
        if ($(item).children().length > 0) {
            $(item).children()[1].remove()
            $(item).children()[1].remove()
            $(item).children()[5].remove()
            $(item).children()[$(item).children().length - 1].remove()
        }
    }
    $('.buttonGroup').on('click', (e) => {
        // console.log(e.target.tagName)
        if (e.target.tagName !== 'BUTTON') {
            return
        } else if (e.target.value === 'ALL') {
            $('#tabResults').children().show()
        } else {
            $('#tabResults').children().hide()
            $('#' + e.target.value).parent().nextUntil('.th').show()
            $('#' + e.target.value).parent().show()
        }
    })
    $('#myTable').DataTable({
        "pageLength": 50,
        "lengthChange": false
    });
});