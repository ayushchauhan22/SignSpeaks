<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Help - SignSpeaks</title>
  <link href="https://unpkg.com/boxicons@2.1.4/css/boxicons.min.css" rel="stylesheet"/>
  <link href="https://fonts.googleapis.com/css2?family=Pacifico&display=swap" rel="stylesheet"/>
  <style>
    html {
      background-image: url('background5.jpg');
      background-repeat: no-repeat;
      background-size: cover;
    }

    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
      text-decoration: none;
    }

    body {
      font-family: Arial, sans-serif;
      padding-top: 90px;
      color: white;
      text-align: center;
      backdrop-filter: blur(10px);
      margin: 0;
    }

    header {
      width: 100%;
      position: fixed;
      top: 0;
      left: 0;
      background: rgba(255, 255, 255, 0.05);
      backdrop-filter: blur(12px);
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
      z-index: 999;
    }

    .navbar {
      max-width: 1200px;
      margin: 0 auto;
      padding: 15px 30px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }

    .title {
      font-size: 36px;
      font-family: 'Pacifico', cursive;
      color: #fff;
      text-shadow: 1px 1px 4px rgba(0, 0, 0, 0.6);
    }

    .nav-links {
      list-style: none;
      display: flex;
      gap: 35px;
    }

    .nav-links a {
      position: relative;
      font-size: 16px;
      font-weight: 500;
      color: #fff;
      text-decoration: none;
    }

    .help-container {
      max-width: 800px;
      margin: auto;
      padding: 40px 20px;
    }

    h2 {
      font-size: 28px;
      margin-bottom: 10px;
    }

    p {
      font-size: 16px;
      margin-bottom: 30px;
      line-height: 1.6;
    }

    form {
      display: flex;
      flex-direction: column;
      align-items: center;
      background-color: rgba(255, 255, 255, 0.1);
      padding: 30px;
      border-radius: 10px;
      box-shadow: 0 0 15px rgba(0,0,0,0.3);
    }

    input, textarea {
      width: 100%;
      max-width: 500px;
      padding: 10px;
      margin: 10px 0;
      border: none;
      border-radius: 8px;
    }

    input[type="submit"] {
      background-color: #ffffff;
      color: #000;
      cursor: pointer;
      font-weight: bold;
      transition: background 0.3s;
    }

    input[type="submit"]:hover {
      background-color: #ddd;
    }
  </style>
</head>
<body>

  <header>
    <div class="navbar">
      <a href="index.html"><h1 class="title">SignSpeaks</h1></a>
      <nav>
        <ul class="nav-links">
          <li><a href="index.html">Home</a></li>
          <li><a href="camera.html">Camera</a></li>
          <li><a href="learningArea.html">Learning Area</a></li>
          <li><a href="help.php">Help</a></li>
          <li><a href="#">About</a></li>
        </ul>
      </nav>
    </div>
  </header>

  <div class="help-container">
    <h2>Facing Issue/Have Suggestions?</h2>
    <p>
      We would love to hear from you! If you have any ideas, feedback, or suggestions to improve SignSpeaks, please use the form below to contact us. 
      If you are facing any issue using the site or have an idea, just send us a message and help us improve!
    </p>

    <form method="POST" action="">
      <input type="text" name="name" placeholder="Your Name" required/>
      <input type="email" name="email" placeholder="Your Email" required/>
      <textarea name="message" rows="10" placeholder="Your Message or Suggestion" required></textarea>
      <input type="submit" value="Send Feedback"/>
    </form>
  </div>

  <?php
  if ($_SERVER["REQUEST_METHOD"] === "POST") {
      $to = "ayufer9@gmail.com"; // Change to your email
      $subject = "New Feedback from SignSpeaks";

      // Sanitize input
      $name = htmlspecialchars($_POST["name"]);
      $email = htmlspecialchars($_POST["email"]);
      $message = htmlspecialchars($_POST["message"]);

      $body = "Name: $name\nEmail: $email\n\nMessage:\n$message";

      $headers = "From: $email\r\nReply-To: $email\r\n";

      if (mail($to, $subject, $body, $headers)) {
          echo "<script>alert('Thank you for your feedback!');</script>";
      } else {
          echo "<script>alert('Sorry, your message could not be sent. Please try again later.');</script>";
      }
  }
  ?>

</body>
</html>
