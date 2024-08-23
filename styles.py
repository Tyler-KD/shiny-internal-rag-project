from shiny import ui

styles_app = ui.page_fluid(
    ui.tags.style("""
        html {
            overflow: hidden !important;
        }
        body {
            overflow: hidden !important;
            height: fit-content; #makes sure the background image isn't cut off
        }
        .row {
          overflow-x: hidden;
          --bs-gutter-x: 0;
        }
        #blueGlobeBG img {
            object-fit: contain !important;
            overflow: hidden !important;
            width: 100vw !important;
            height: auto !important;
        }
        #blueGlobeBG {
            position: absolute;
            top: 0;
            left: 0;
        }
        .bslib-sidebar-layout {
            --_sidebar-width: 25vw !important; /* Override the sidebar width to 1/4 of the screen */
            # background-image: url('https://www.lixcap.com/wp-content/uploads/section-2-bg.png');
            # background-size: cover;
            # background-position: center;
            # background-repeat: no-repeat;
        }
        .bslib-sidebar-layout>.sidebar {
            border-right: 0;
            backdrop-filter: blur(3px);
            background: rgba(255, 255, 255, 0.07);
        }
        .bslib-sidebar-layout>.sidebar>.sidebar-content {
            padding: 24px;
            gap: 0;
        }
        #file_info {
            margin: 20px 0;
        }
        .form-group,.form-check, .shiny-input-container .checkbox, .shiny-input-container .radio {
            margin: 0;
        }
        #fixedFileSpacing>.mb-2,uploaded-files-spacing {
            margin-bottom: 18px !important;
        }
        .checkbox label {
            width: 100%;
            cursor: default !important;
        }
        .checkbox label input {
            cursor: pointer;         
        }
        .accordion-body {
            padding: 8px 20px 0;
        }
        .sidebar {
            height: 100vh;
            box-shadow: 2px 0 5px rgba(0,0,0,0.2);
            overflow-y: auto;
        }
        .sidebar::-webkit-scrollbar {
            width: 10px;
        }
        .sidebar::-webkit-scrollbar-track {
            background: #0E4878;
        }
        .sidebar::-webkit-scrollbar-thumb {
            background: #589FD5;
        }
        .btn-primary {
            background-color: #0E4878;
            border-color: #0E4878;
            color: #FFFFFF;
        }
        .btn-primary:hover, .file-input:hover::file-selector-button {
            background-color: #589FD5;
            border-color: #589FD5;
        }
        .btn-danger {
            background-color: #C14B59;
            border-color: #C14B59;
            color: #FFFFFF;
        }
        .btn-danger:hover {
            background-color: #A83240;
            border-color: #A83240;
        }
        .form-control:focus {
            border-color: #589FD5;
            box-shadow: 0 0 0 0.2rem rgba(88, 159, 213, 0.25);
        }
        #process_output {
            background-color: #F5F5F5;
            border: 1px solid #E0E0E0;
            padding: 15px;
            margin-top: 20px;
            border-radius: 7px;
            white-space: pre-wrap;
        }
        h1, h2, h3, h4, h5, li {
            color: #0E4878;
        }
        .mt-2 {
            margin-top: 0.5rem;
        }
        .mt-3 {
            margin-top: 1rem;
        }
        .sidebar .form-check-label {
            color: #4E4E4E;
        }
        .logoWelcome {
            display: flex;
            justify-content: space-between;
        }
        #time_display {
            text-align: right;
        }
        .time-display-column {
            font-size: 1.1em;
            display: flex;
            flex-direction: column;
            jusify-content: space-between;
        }
        .time-display-column {
            justify-content: space-between;
        }
        .time-display-column > h1.greeting {
            font-size: 2.5em;
            font-weight: bold;
            font-style: italic;
            margin: 0;
        }
        # .language-toggle {
        #     margin-top: 20px;
        #     margin-bottom: 20px;
        # }
        .file-input::file-selector-button {
            background-color: #0E4878;
            color: #FFFFFF;
            border: none;
            padding: 5px 10px;
            border-radius: 3px;
            cursor: pointer;
        }
        .shiny-input-container:not(.shiny-input-container-inline) {
            width: 100%;
        }
        .sensitive-file {
            color: red;
            font-weight: bold;
        }
    """)
)