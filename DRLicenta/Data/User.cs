namespace DRLicenta.Data
{
    public class User
    {
        public int Id { get; set; }

        public required string CNP { get; set; }

        public required string Email { get; set; }

        public required string PhoneNumber { get; set; }

        public required string PasswordHash { get; set; }

        public required string PasswordSalt { get; set;}
    }
}
