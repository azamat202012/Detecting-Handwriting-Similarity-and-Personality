import match

def main():
	function = match.MatchImages()

	value = function.matchLight("./Images/Handwriting/8.jpeg", "./Images/Handwriting/1.jpeg")
	print(value)

if __name__ == "__main__":
	main()
