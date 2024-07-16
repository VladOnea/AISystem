export class GlobalConstants {
  //Message
  public static genericError: string =
    'Something went wrong. Please try again later!';

  public static unauthorized: string =
    'You are not authorized person to access this page!';

  public static error: string = 'Error!';

  //Regex
  public static CNPRegex: string =
    '^(1|2|3|4|5|6|7|8|9)\\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\\d|3[01])\\d{2}\\d{3}\\d$';

  public static emailRegex: string =
    '[A-Za-z0-9._%-]+@[A-Za-z0-9._%-]+\\.[a-z]{2,3}';

  public static telephoneNumberRegex: string = '^[0-9]{10}$';
}
