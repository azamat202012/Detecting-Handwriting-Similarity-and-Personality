import match

def main():
	function = match.MatchImages()

	value = function.matchDeep("./Images/Signature/7.jpeg", "./Images/Handwriting/7.jpeg")
	print(value)

if __name__ == "__main__":
	main()
