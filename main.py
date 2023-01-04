import match

def main():
	function = match.MatchImages()

	value = function.matchDeep("./Images/Handwriting/8.jpeg", "./Images/Handwriting/2.jpeg")
	print(value)

if __name__ == "__main__":
	main()
