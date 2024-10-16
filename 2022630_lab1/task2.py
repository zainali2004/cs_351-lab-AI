def ai_number_guessing_game():
    # Player selects a number
    print("Think of a number between 1 and 100, and I (the AI) will try to guess it.")
    low = 1
    high = 100
    attempts = 0

    # Loop until the AI guesses the number correctly
    while low <= high:
        guess = (low + high) // 2  # AI uses binary search to guess
        attempts += 1

        print(f"AI's guess is: {guess}")
        feedback = input("Enter 'h' if too high, 'l' if too low, or 'c' if correct: ").lower()

        if feedback == 'c':
            print(f"I (AI) guessed the number in {attempts} attempts!")
            return
        elif feedback == 'h':
            high = guess - 1  # If too high, reduce the upper bound
        elif feedback == 'l':
            low = guess + 1  # If too low, increase the lower bound

    print("Something went wrong!")

# Run the AI version
ai_number_guessing_game()