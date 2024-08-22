from shiny import ui

styles_app = ui.page_fluid(
    ui.tags.style("""
        body {
            font-family: Arial, sans-serif;
            background-color: #E0E0E0;
            color: #4E4E4E;
            overflow-y: hidden;
        }
        .row {
          overflow-x: hidden;
          --bs-gutter-x: 0;
        }
        .sidebar {
            background-color: #FFFFFF;
            color: #0E4878;
            padding: 20px;
            height: 100vh;
            box-shadow: 2px 0 5px rgba(0,0,0,0.1);
            overflow-y: auto;
            max-width: 300px;
        }
        .main-content {
            # background-image: url("/whitelogo.jpg");
            # background-size: cover;
            # background-position: center;
            # background-repeat: no-repeat;
            background-color: #FFFFFF;
            padding: 20px;
            height: 100vh;
            overflow-x: hidden;
            position: relative;
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
            border-radius: 5px;
            white-space: pre-wrap;
        }
        h1, h2, h3 {
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
        .time-display {
            font-size: 0.9em;
            text-align: right;
            margin-bottom: 15px;
        }
        .greeting {
            font-size: 1.5em;
            font-weight: bold;
            font-style: italic;
            margin-bottom: 16px;
        }
        .language-toggle {
            margin-top: 20px;
            margin-bottom: 20px;
        }
        .logo {
            position: absolute;
            top: 20px;
            left: 20px;
            max-width: 150px;
            height: auto;
            z-index: 1000;
        }
        .logo-error {
            color: red;
            font-style: italic;
        }
        .file-input::file-selector-button {
            background-color: #0E4878;
            color: #FFFFFF;
            border: none;
            padding: 5px 10px;
            border-radius: 3px;
            cursor: pointer;
        }
        .sensitive-file {
            color: red;
            font-weight: bold;
        }
    """)
)