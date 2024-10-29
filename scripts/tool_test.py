from mplsandbox import MPLSANDBOX
import os
import json

if __name__ == '__main__':

    # Data_rust = {
    # "code": "use std::io::{self, Read};\n\n// 快速排序函数\nfn quick_sort(arr: &mut [i32]) {\n    if arr.len() <= 1 {\n        return;\n    }\n\n    let pivot = arr[arr.len() / 2];\n    let (mut left, mut right) = (0, arr.len() - 1);\n\n    while left <= right {\n        while arr[left] < pivot {\n            left += 1;\n        }\n        while arr[right] > pivot {\n            right = right.checked_sub(1).unwrap_or(0);\n        }\n\n        if left <= right {\n            arr.swap(left, right);\n            left += 1;\n            if right == 0 {\n                break; // 避免溢出\n            }\n            right -= 1;\n        }\n    }\n\n    if right > 0 {\n        quick_sort(&mut arr[0..=right]);\n    }\n    if left < arr.len() {\n        quick_sort(&mut arr[left..]);\n    }\n}\n\nfn main() {\n    let mut input = String::new();\n    io::stdin().read_to_string(&mut input).expect(\"Failed to read line\");\n\n    let mut numbers: Vec<i32> = input.split_whitespace()\n        .map(|num| num.parse().expect(\"Please type a number!\"))\n        .collect();\n\n    quick_sort(&mut numbers);\n\n    println!(\"Sorted numbers: {:?}\", numbers);\n}\n",
    # "unit_cases": {
    # "inputs": [
    #         "34 7 23 32 5 62"
    #     ],
    #     "outputs": [
    #         "Sorted numbers: [5, 7, 23, 32, 34, 62]"
    #     ]
    # },
    # "lang": "rust"}

    # Data_python = {
    # "code": "a=input()\nb=input()\nprint(int(a)+int(b))",
    # "unit_cases": {
    # "inputs": ["1\n2", "3\n4"],
    # "outputs": ["3", "7"]
    # },
    # "lang": "python"
    # }

    # Data_cpp={
    #   "code": "#include<iostream>\nusing namespace std;\nint main(){\n\tint a,b;\n\tcin>>a>>b;\n\tcout<<a+b;\n\treturn 0;\n}",
    #   "unit_cases": {
    #     "inputs": ["1 2", "3 4"],
    #     "outputs": ["3", "7"]
    #   },
    #   "lang": "cpp"
    # }


    Data_java={
      "question":"Define get_sum_of_two_numbers():\n    \"\"\"Write a function that takes two integers as input and returns their sum.\n\n    -----Input-----\n    \n    The input consists of multiple test cases. Each test case contains two integers $a$ and $b$ ($-10^9 \\le a, b \\le 10^9$).\n    \n    -----Output-----\n    \n    For each test case, print the sum of the two integers.\n    \n    -----Example-----\n    Input\n    3\n    1 2 ↵\n    -1 1 ↵\n    1000000000 1000000000\n    \n    Output\n    3\n    0\n    2000000000\n    \"\"\"",
      "code":"import java.util.Scanner;\npublic class Main{\n\tpublic static void main(String[] args){\n\t\tScanner scanner = new Scanner(System.in);\n\t\tint a = scanner.nextInt();\n\t\tint b = scanner.nextInt();\n\t\tSystem.out.println(a+b);\n\t}\n}",
      "unit_cases": {
        "inputs": ["1 2", "3 4"],
        "outputs": ["3", "7"]
      },
      "lang": "java"
    }

    Data_go = {
    "question":"Define get_sum_of_two_numbers():\n    \"\"\"Write a function that takes two integers as input and returns their sum.\n\n    -----Input-----\n    \n    The input consists of multiple test cases. Each test case contains two integers $a$ and $b$ ($-10^9 \\le a, b \\le 10^9$).\n    \n    -----Output-----\n    \n    For each test case, print the sum of the two integers.\n    \n    -----Example-----\n    Input\n    3\n    1 2 ↵\n    -1 1 ↵\n    1000000000 1000000000\n    \n    Output\n    3\n    0\n    2000000000\n    \"\"\"",    
      "code":"package main\n\nimport (\n\t\"fmt\"\n)\n\nfunc main() {\n\tvar a, b int\n\tfmt.Scanf(\"%d %d\", &a, &b)\n\tfmt.Printf(\"%d\\n\", a+b)\n}",
      "unit_cases": {
        "inputs": ["1 2", "3 4"],
        "outputs": ["3", "7"]
      },
      "lang": "go"
    }



    # Data_javascript = {
    #  "code": "const readline = require('readline');\n\nconst rl = readline.createInterface({\n    input: process.stdin,\n    output: process.stdout\n});\n\nrl.on('line', (input) => {\n    const numbers = input.split(' ').map(Number);\n    quickSort(numbers, 0, numbers.length - 1);\n    console.log(numbers.join(' '));\n    rl.close();\n});\n\nfunction quickSort(arr, left, right) {\n    if (left < right) {\n        const partitionIndex = partition(arr, left, right);\n        quickSort(arr, left, partitionIndex - 1);\n        quickSort(arr, partitionIndex + 1, right);\n    }\n}\n\nfunction partition(arr, left, right) {\n    const pivot = arr[right];\n    let i = left - 1;\n    for (let j = left; j < right; j++) {\n        if (arr[j] <= pivot) {\n            i++;\n            swap(arr, i, j);\n        }\n    }\n    swap(arr, i + 1, right);\n    return i + 1;\n}\n\nfunction swap(arr, i, j) {\n    [arr[i], arr[j]] = [arr[j], arr[i]];\n}\n",
    #  "unit_cases": {
    #   "inputs": [
    #    "5 3 8 6 2 7 4 1",
    #    "8 3 7 4 9 2 6 5"
    #   ],
    #   "outputs": [
    #    "1 2 3 4 5 6 7 8",
    #    "2 3 4 5 6 7 8 9"
    #   ]
    #  },
    #  "lang": "javascript"
    # }

    # Data_python2 = {   
    # "question":"Define get_sum_of_two_numbers():\n    \"\"\"Write a function that takes two integers as input and returns their sum.\n\n    -----Input-----\n    \n    The input consists of multiple test cases. Each test case contains two integers $a$ and $b$ ($-10^9 \\le a, b \\le 10^9$).\n    \n    -----Output-----\n    \n    For each test case, print the sum of the two integers.\n    \n    -----Example-----\n    Input\n    3\n    1 2 ↵\n    -1 1 ↵\n    1000000000 1000000000\n    \n    Output\n    3\n    0\n    2000000000\n    \"\"\"",
    # "code": "def get_sum_of_two_numbers():\n    a=input()\n    b=input()\n    print(int(a)*int(b))\nget_sum_of_two_numbers()\n\n\n",
    # "unit_cases": {
    # "inputs": ["1\n2", "3\n4"],
    # "outputs": ["3", "7"]
    # },
    # "lang": "python"
    # }

    Data_python = {   
    "question":"Define get_sum_of_two_numbers():\n    \"\"\"Write a function that takes two integers as input and returns their sum.\n\n    -----Input-----\n    \n    The input consists of multiple test cases. Each test case contains two integers $a$ and $b$ ($-10^9 \\le a, b \\le 10^9$).\n    \n    -----Output-----\n    \n    For each test case, print the sum of the two integers.\n    \n    -----Example-----\n    Input\n    3\n    1 2 ↵\n    -1 1 ↵\n    1000000000 1000000000\n    \n    Output\n    3\n    0\n    2000000000\n    \"\"\"",
    "code": 'def get_sum_of_two_numbers():\n    a, b = map(int, input().split(" "))\n    print(a * b)\nget_sum_of_two_numbers()',
    "unit_cases": {
    "inputs": ["1 2", "3 4"],
    "outputs": ["3", "7"]
    },
    "lang": "python"
    }


    executor = MPLSANDBOX(Data_python)
    res = executor.run("all")
    print(res)


    

