# FastOCR, FastCalculator, ImageDescriptor, WebSearch

tool_description = '''

\n[{'name': Calculator', 'description': 'A calculator tool. The input must be a single Python expression and you cannot import packages. You can use functions in the `math` package without import.',
'parameters': [{'name': 'expression', 'description': None, 'type': 'STRING'}],
'required': ['expression'],'parameter_description': 'If you call this tool, you must pass arguments in the JSON format {key: value}, where the key is the parameter name.'}]

\n[{'name': FastCalculator', 'description': 'A calculator tool. The input must be a single Python expression and you cannot import packages. You can use functions in the `math` package without import.',
'parameters': [{'name': 'expression', 'description': None, 'type': 'STRING'}],
'required': ['expression'],'parameter_description': 'If you call this tool, you must pass arguments in the JSON format {key: value}, where the key is the parameter name.'}]

\n[{'name': 'OCR', 'description': 'This tool can recognize all text on the input image.',
'parameters': [{'name': 'image', 'description': None, 'type': 'STRING'}],
'required': ['image'],'parameter_description': 'If you call this tool, you must pass arguments in the JSON format {key: value}, where the key is the parameter name.'}]

\n[{'name': 'FastOCR', 'description': 'This tool can recognize all text on the input image.',
'parameters': [{'name': 'image', 'description': None, 'type': 'STRING'}], 
'required': ['image'],'parameter_description': 'If you call this tool, you must pass arguments in the JSON format {key: value}, where the key is the parameter name.'}]

\n[{'name': 'CountGivenObject', 'description': 'The tool can count the number of a certain object in the image.',
'parameters': [{'name': 'image', 'description': None, 'type': 'STRING'}, {'name': 'text', 'description': 'The object description in English.', 'type': 'STRING'}], 
'required': ['image', 'text'],'parameter_description': 'If you call this tool, you must pass arguments in the JSON format {key: value}, where the key is the parameter name.'}]

\n[{'name': 'ImageDescription', 'description': 'A useful tool that returns a brief description of the input image.',
'parameters': [{'name': 'image', 'description': None, 'type': 'STRING'}],
'required': ['image'],'parameter_description': 'If you call this tool, you must pass arguments in the JSON format {key: value}, where the key is the parameter name.'}]

\n[{'name': 'ImageDescriptor', 'description': 'A useful tool that returns a brief description of the input image.',
'parameters': [{'name': 'image', 'description': None, 'type': 'STRING'}],
'required': ['image'],'parameter_description': 'If you call this tool, you must pass arguments in the JSON format {key: value}, where the key is the parameter name.'}]

\n[{'name': 'GoogleSearch', 'description': 'The tool can search the input query text from Google and return the related results.',
'parameters': [{'name': 'query', 'description': None, 'type': 'STRING'}, {'name': 'k', 'description': 'Select the first k results', 'type': 'INT'}],
'required': ['query'],'parameter_description': 'If you call this tool, you must pass arguments in the JSON format {key: value}, where the key is the parameter name.'}]

\n[{'name': 'WebSearch', 'description': 'The tool can search the input query text from Google and return the related results.',
'parameters': [{'name': 'query', 'description': None, 'type': 'STRING'}, {'name': 'k', 'description': 'Select the first k results', 'type': 'INT'}],
'required': ['query'],'parameter_description': 'If you call this tool, you must pass arguments in the JSON format {key: value}, where the key is the parameter name.'}]

\n[{'name': 'TextToBbox', 'description': 'The tool can detect the object location according to description.',
'parameters': [{'name': 'image', 'description': None, 'type': 'STRING'}, {'name': 'text', 'description': 'The object description in English.', 'type': 'STRING'}, {'name': 'top1', 'description': 'If true, return the object with highest score. If false, return all detected objects.', 'type': 'BOOL'}],
'required': ['image', 'text'],'parameter_description': 'If you call this tool, you must pass arguments in the JSON format {key: value}, where the key is the parameter name.'}]

\n[{'name': 'Plot', 'description': 'This tool can execute Python code to plot diagrams. The code should include a function named \'solution\'. The function should return the matplotlib figure directly. Avoid printing the answer. The code instance format is as follows:\n\n```python\n# import packages\nimport matplotlib.pyplot as plt\ndef solution():\n    # labels and data\n    cars = [\'AUDI\', \'BMW\', \'FORD\', \'TESLA\', \'JAGUAR\', \'MERCEDES\']\n    data = [23, 17, 35, 29, 12, 41]\n\n    # draw diagrams\n    figure = plt.figure(figsize=(8, 6))\n    plt.pie(data, labels=cars, autopct=\'%1.1f%%\', startangle=140)\n    plt.axis(\'equal\')\n    plt.title(\'Car Distribution\')\n    return figure\n```',
'parameters': [{'name': 'command', 'description': 'Markdown format Python code', 'type': 'STRING'}],
'required': ['command'],'parameter_description': 'If you call this tool, you must pass arguments in the JSON format {key: value}, where the key is the parameter name.'}]

\n[{'name': 'MathOCR', 'description': 'This tool can recognize math expressions from an image and return the latex style expression.',
'parameters': [{'name': 'image', 'description': None, 'type': 'STRING'}],
'required': ['image'],'parameter_description': 'If you call this tool, you must pass arguments in the JSON format {key: value}, where the key is the parameter name.'}]

\n[{'name': 'Solver', 'description': 'This tool can execute Python code to solve math equations. The code should include a function named 'solution'. You should use the `sympy` library in your code to solve the equations. The function should return its answer in str format. Avoid printing the answer. The code instance format is as follows:\n\n```python\n# import packages\nfrom sympy import symbols, Eq, solve\ndef solution():\n    # Define symbols\n    x, y = symbols(\'x y\')\n\n    # Define equations\n    equation1 = Eq(x**2 + y**2, 20)\n    equation2 = Eq(x**2 - 5*x*y + 6*y**2, 0)\n\n    # Solve the system of equations\n    solutions = solve((equation1, equation2), (x, y), dict=True)\n\n    # Return solutions as strings\n    return str(solutions)\n```',
'parameters': [{'name': 'command', 'description': 'Markdown format Python code', 'type': 'STRING'}],
'required': ['command'],'parameter_description': 'If you call this tool, you must pass arguments in the JSON format {key: value}, where the key is the parameter name.'}]

\n[{'name': 'DrawBox', 'description': 'A tool to draw a box on a certain region of the input image.',
'parameters': [{'name': 'image', 'description': None, 'type': 'STRING'}, {'name': 'bbox', 'description': 'The bbox coordinate in the format of `(x1, y1, x2, y2)`', 'type': 'STRING'}, {'name': 'annotation', 'description': 'The extra annotation text of the bbox', 'type': 'STRING'}],
'required': ['image', 'bbox'],'parameter_description': 'If you call this tool, you must pass arguments in the JSON format {key: value}, where the key is the parameter name.'}]

\n[{'name': 'AddText', 'description': 'A tool to draw a box on a certain region of the input image.',
'parameters': [{'name': 'image', 'description': None, 'type': 'STRING'}, {'name': 'text', 'description': None, 'type': 'STRING'}, {'name': 'position', 'description': 'The left-bottom corner coordinate in the format of `(x, y)`, or a combination of [\"l\"(left), \"m\"(middle), \"r\"(right)] and [\"t\"(top), \"m\"(middle), \"b\"(bottom)] like \"mt\" for middle-top', 'type': 'STRING'}, {'name': 'color', 'description': None, 'type': 'STRING'}],
'required': ['image', 'text', 'position'],'parameter_description': 'If you call this tool, you must pass arguments in the JSON format {key: value}, where the key is the parameter name.'}]

\n[{'name': 'TextToImage', 'description': 'This tool can generate an image according to the input text.',
'parameters': [{'name': 'keywords', 'description': 'A series of keywords separated by comma.', 'type': 'STRING'}],
'required': ['keywords'],'parameter_description': 'If you call this tool, you must pass arguments in the JSON format {key: value}, where the key is the parameter name.'}]

\n[{'name': 'ImageStylization', 'description': 'This tool can modify the input image according to the input instruction. Here are some example instructions: \"turn him into cyborg\", \"add fireworks to the sky\", \"make his jacket out of leather\".',
'parameters': [{'name': 'image', 'description': None, 'type': 'STRING'}, {'name': 'instruction', 'description': None, 'type': 'STRING'}],
'required': ['image', 'instruction'],'parameter_description': 'If you call this tool, you must pass arguments in the JSON format {key: value}, where the key is the parameter name.'}]
'''

action_names = 'Calculator, FastCalculator, FastOCR, OCR, CountGivenObject, ImageDescription, GoogleSearch, WebSearch, TextToBbox, Plot, MathOCR, Solver, DrawBox, AddText, TextToImage, ImageStylization'