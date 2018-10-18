PrintWriter output;
String currColour = "";
float red = random(1);
float green = random(1);
float blue = random(1);
int count = 0;

void setup()
{
  output = createWriter("testing.txt");
  size(640,360);
}
void draw(){
  background(red*255,green*255,blue*255);
  fill(0);
  textSize(12);
  text(currColour,20,20);
  text("Choose from red, orange, yellow, green, blue, purple, black, grey, or white.",20,40);
  text("Press enter to enter your guess and press backspace to delete your guess if you spelled something incorrectly.",20,60);
  text("Press escape to quit.",20,80);
  text(count, 20,100);
}
void keyPressed()
{
  if (key != ESC && key != ENTER && key != BACKSPACE)
  {
    currColour += key;
  }
  else if (key == ESC)
  {
    output.flush();
    output.close();
    exit();
  }
  else if (key == BACKSPACE)
  {
    currColour = "";
  }
  else if (key == ENTER)
  {
    output.println(red+" "+green+" "+blue+" "+currColour); // all credit goes to Daniel for figuring out the problem here
    red = random(1);
    green = random(1);
    blue = random(1);
    currColour = "";
    count++;
  }
}
