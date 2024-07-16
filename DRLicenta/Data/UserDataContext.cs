using Microsoft.EntityFrameworkCore;


namespace DRLicenta.Data
{
    public class UserDataContext : DbContext
    {
        public UserDataContext(DbContextOptions options) : base(options)
        {

        }

        public DbSet<User> Users { get; set; }

        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
            base.OnModelCreating(modelBuilder);
        }
    }
}
