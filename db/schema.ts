import { Pool } from 'pg';

const pool = new Pool({
    connectionString: process.env.RAILWAY_DATABASE_URL,
    ssl: {
        rejectUnauthorized: false
    }
});

export const initializeDatabase = async () => {
    try {
        await pool.query(`
            CREATE TABLE IF NOT EXISTS addresses (
                id SERIAL PRIMARY KEY,
                raw_address TEXT NOT NULL,
                normalized_address TEXT NOT NULL,
                components JSONB NOT NULL DEFAULT '{}'::jsonb,
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_normalized_address 
            ON addresses(normalized_address);
            
            CREATE INDEX IF NOT EXISTS idx_postal_code 
            ON addresses((components->>'postal_code'));
        `);

        console.log('Database initialized successfully');
    } catch (error) {
        console.error('Failed to initialize database:', error);
        throw error;
    }
};

export { pool };